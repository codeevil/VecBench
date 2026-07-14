"""
pgvector_llm_optimizer.py

Single-file pgvector filtered vector query optimizer.

Features:
- Collects PostgreSQL / pgvector catalog stats before planning.
- Produces safe PlanSpecs for filtered vector similarity queries.
- Supports target modes: absolute_recall, recall_first, performance_first.
- Prevents recall disasters for HNSW + WHERE by explicitly setting iterative scan and ef_search.
- Extensible: add new rules with @register_rule; add prompt fragments with @register_prompt_fragment.

Phases:
  init — Generate candidates (rule-based or LLM) and save to a JSON file.
  test — Load candidates from file, benchmark, print compact recall+latency.

Usage:
  # init: generate plans, save to file
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --yaml-queries config/SIFT/E2E_queries_1M.yaml \
  --schema-config config/SIFT/schema.yaml \
  --table-name my_table \
  --distance-ops vector_l2_ops \
  --mode balanced \
  --candidates 10 \
  --output /home/liujianzhong/VecBench/baseline/vecLLM/llm_candidates.json \
  --llm \
  --query-name query1


  # init + llm: use LLM to generate candidates
  python pgvector_llm_optimizer.py init --llm \
    --yaml-queries config/SIFT/100.E2E_queries_1M.yaml \
    --schema-config config/SIFT/schema.yaml \
    --distance-ops vector_l2_ops \
    --mode recall_first --candidates 5 \
    --output llm_candidates.json

  # test: benchmark candidates and print results
python baseline/vecLLM/pgvector_llm_optimizer.py test \
  --input /home/liujianzhong/VecBench/baseline/vecLLM/llm_candidates.json \
  --ground-truth data/SIFT/ground_truth/E2E_1M_128.json \
  --warmup 0 --runs 10

  python pgvector_llm_optimizer.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Literal, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import psycopg
except ImportError:
    psycopg = None

TargetMode = Literal["absolute_recall", "recall_first", "performance_first", "balanced"]
RuleFn = Callable[["RuleContext"], Optional[dict[str, Any]]]
PromptFragmentFn = Callable[["OptimizerConfig"], str]

RULES: list[RuleFn] = []
PROMPT_FRAGMENTS: list[PromptFragmentFn] = []


def register_rule(fn: RuleFn) -> RuleFn:
    """Add a new planning rule. Rules are checked in registration order."""
    RULES.append(fn)
    return fn


def register_prompt_fragment(fn: PromptFragmentFn) -> PromptFragmentFn:
    """Add a new LLM prompt fragment."""
    PROMPT_FRAGMENTS.append(fn)
    return fn


@dataclass
class OptimizerConfig:
    target_mode: TargetMode = "recall_first"
    candidate_count: int = 3
    min_acceptable_recall: float = 0.70
    high_recall_target: float = 0.95
    max_latency_ms: int | None = 200
    index_budget: Literal["no_new_indexes", "allow_new_indexes", "allow_partial_indexes"] = "allow_new_indexes"
    default_schema: str = "public"
    exact_scan_threshold_small: int = 50_000
    exact_scan_threshold_medium: int = 20_000
    exact_scan_threshold_large: int = 10_000
    model: str = "DeepSeek-V4-Flash"


@dataclass
class QueryShape:
    schema: str
    table: str
    where_clause: str | None
    order_by: str | None
    limit_k: int
    vector_column: str | None
    metric_operator: str | None


@dataclass
class RuleContext:
    sql: str
    config: OptimizerConfig
    shape: QueryShape
    stats: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    missing_stats: list[str] = field(default_factory=list)

    @property
    def indexes(self) -> list[dict[str, Any]]:
        return self.stats.get("indexes", []) or []

    @property
    def estimated_rows(self) -> int:
        return int(self.stats.get("estimated_rows") or 0)

    @property
    def filtered_rows(self) -> int | None:
        v = self.stats.get("filtered_rows_estimate")
        return int(v) if v is not None else None

    @property
    def selectivity(self) -> float | None:
        v = self.stats.get("estimated_selectivity")
        return float(v) if v is not None else None


def parse_query_shape(sql: str, default_schema: str = "public") -> QueryShape:
    compact = " ".join(sql.strip().rstrip(";").split())
    from_m = re.search(r"\bFROM\s+([\w\"]+(?:\.[\w\"]+)?)", compact, re.I)
    if not from_m:
        raise ValueError("Cannot find FROM table in SQL")

    table_ref = from_m.group(1).replace('"', "")
    if "." in table_ref:
        schema, table = table_ref.split(".", 1)
    else:
        schema, table = default_schema, table_ref

    where_clause = None
    where_m = re.search(r"\bWHERE\s+(.+?)(\bORDER\s+BY\b|\bLIMIT\b|$)", compact, re.I)
    if where_m:
        where_clause = where_m.group(1).strip()

    order_by = None
    order_m = re.search(r"\bORDER\s+BY\s+(.+?)(\bLIMIT\b|$)", compact, re.I)
    if order_m:
        order_by = order_m.group(1).strip()

    limit_k = 10
    limit_m = re.search(r"\bLIMIT\s+(\d+)", compact, re.I)
    if limit_m:
        limit_k = int(limit_m.group(1))

    vector_column = None
    metric_operator = None
    if order_by:
        metric_m = re.search(r"([\w\"]+)\s*(<->|<=>|<#>|<\+>|<~>|<%>)", order_by)
        if metric_m:
            vector_column = metric_m.group(1).replace('"', "")
            metric_operator = metric_m.group(2)

    return QueryShape(schema, table, where_clause, order_by, limit_k, vector_column, metric_operator)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_regclass(schema: str, table: str) -> str:
    return f"{qident(schema)}.{qident(table)}"


def clamp(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(math.ceil(value))))


def has_index(indexes: list[dict[str, Any]], keyword: str) -> bool:
    k = keyword.lower()
    return any(k in str(idx.get("indexdef", "")).lower() for idx in indexes)


def metric_ops(metric_operator: str | None) -> str:
    return {
        "<=>": "vector_cosine_ops",
        "<->": "vector_l2_ops",
        "<#>": "vector_ip_ops",
        "<+>": "vector_l1_ops",
    }.get(metric_operator or "", "vector_cosine_ops")


# pgvector distance operator from ops name (reverse of metric_ops)
DISTANCE_OPS_TO_OPERATOR: dict[str, str] = {
    "vector_l2_ops": "<->",
    "vector_cosine_ops": "<=>",
    "vector_ip_ops": "<#>",
    "vector_l1_ops": "<+>",
}


def yaml_query_to_sql(
    query: dict[str, Any],
    table_name: str,
    distance_ops: str = "vector_cosine_ops",
) -> str:
    """Convert a VecBench-style query dict into a pgvector SQL string.

    Parameters
    ----------
    query : dict
        VecBench query dict with keys: vector_field, reference_vector_name,
        scalar_filters (list of {field, operator, value, logic}), limit.
    table_name : str
        The database table name (e.g. "my_table").
    distance_ops : str
        pgvector operator class name (e.g. "vector_l2_ops", "vector_cosine_ops").

    Returns
    -------
    str
        A pgvector ANN SQL query string.
    """
    vector_field = query["vector_field"]
    reference_vector_name = query["reference_vector_name"]
    scalar_filters = query.get("scalar_filters", [])
    limit = query["limit"]

    distance_op = DISTANCE_OPS_TO_OPERATOR.get(distance_ops, "<=>")

    # Build WHERE clause from scalar_filters
    scalar_conditions: list[tuple[str, str]] = []
    for f in scalar_filters:
        field = f["field"]
        operator = f["operator"]
        value = f["value"]
        logic = f.get("logic", "and")

        if operator == "==":
            condition = f"{field} = {value}"
        elif operator == "<":
            condition = f"{field} < '{value}'"
        elif operator == "<=":
            condition = f"{field} <= '{value}'"
        elif operator == ">":
            condition = f"{field} > '{value}'"
        elif operator == ">=":
            condition = f"{field} >= '{value}'"
        elif operator == "like":
            condition = f"{field} LIKE '%{value}%'"
        elif operator == "contains":
            condition = f"'{value}' = ANY({field})"
        else:
            continue

        scalar_conditions.append((condition, logic))

    if scalar_conditions:
        where_str = scalar_conditions[0][0]
        for condition, logic in scalar_conditions[1:]:
            where_str += f" {logic} {condition}"
        where_str = f"WHERE {where_str}"
    else:
        where_str = ""

    sql = (
        f"SELECT id FROM {table_name} "
        f"{where_str} "
        f"ORDER BY ({vector_field} {distance_op} "
        f"(SELECT {vector_field} FROM {table_name} WHERE id = {reference_vector_name})) "
        f"LIMIT {limit}"
    )
    # Normalize whitespace
    return " ".join(sql.strip().split())


def load_vecbench_queries(yaml_path: str, dataset: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Load VecBench query YAML file and return (dataset_name, queries).

    Parameters
    ----------
    yaml_path : str
        Path to the YAML query file (e.g. config/SIFT/100.E2E_queries_1M.yaml).
    dataset : str or None
        Dataset key inside the YAML (e.g. "SIFT"). If None, the first key is used.

    Returns
    -------
    tuple[str, list[dict]]
        The resolved dataset name and the list of query dicts.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Run: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {yaml_path} does not contain a top-level mapping")
    if dataset is None:
        dataset = next(iter(data.keys()))
    if dataset not in data:
        raise KeyError(f"Dataset '{dataset}' not found in {yaml_path}. Available: {list(data.keys())}")
    queries = data[dataset]
    if not isinstance(queries, list):
        raise ValueError(f"Expected a list of queries for dataset '{dataset}'; got {type(queries).__name__}")
    return dataset, queries


def load_schema_config(yaml_path: str, database: str = "pgvector") -> dict[str, Any]:
    """Load VecBench schema.yaml and return the database-specific section."""
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Run: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if database not in data:
        raise KeyError(f"Database '{database}' not found in schema config. Available: {list(data.keys())}")
    return data[database]


def load_app_config(path: str | None) -> dict[str, Any]:
    """Load JSON config. Missing file returns empty dict."""
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dsn_from_config(app_cfg: dict[str, Any], cli_dsn: str | None = None) -> str | None:
    """Resolve DSN from CLI, config.pg.dsn, config.pg fields, then PG_DSN env."""
    if cli_dsn:
        return cli_dsn
    pg = app_cfg.get("pg", {}) if isinstance(app_cfg, dict) else {}
    if pg.get("dsn"):
        return pg["dsn"]
    if pg.get("host") and pg.get("database") and pg.get("user"):
        host = pg.get("host", "localhost")
        port = pg.get("port", 5432)
        db = pg["database"]
        user = pg["user"]
        password = pg.get("password", "")
        # Keep it simple; for special characters prefer pg.dsn in config.json.
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return os.getenv("PG_DSN")


def optimizer_config_from_app(app_cfg: dict[str, Any], mode: str | None = None, candidates: int | None = None) -> OptimizerConfig:
    raw = dict(app_cfg.get("optimizer", {}) or {})
    if mode:
        raw["target_mode"] = mode
    if candidates is not None:
        raw["candidate_count"] = candidates
    allowed = set(OptimizerConfig.__dataclass_fields__.keys())
    raw = {k: v for k, v in raw.items() if k in allowed}
    return OptimizerConfig(**raw)


def size_based_exact_threshold(n: int, cfg: OptimizerConfig) -> int:
    if n <= 100_000:
        return min(n or cfg.exact_scan_threshold_small, cfg.exact_scan_threshold_small)
    if n <= 1_000_000:
        return cfg.exact_scan_threshold_small
    if n <= 100_000_000:
        return cfg.exact_scan_threshold_medium
    return cfg.exact_scan_threshold_large


def validation_sql() -> str:
    return (
        "BEGIN;\n"
        "SET LOCAL enable_indexscan = off;\n"
        "SET LOCAL enable_bitmapscan = off;\n"
        "-- Run exact baseline top-K, compare overlap with optimized top-K.\n"
        "COMMIT;"
    )


def base_plan(ctx: RuleContext, chosen_path: str, recall: float | None, qps: str) -> dict[str, Any]:
    return {
        "version": "1.0",
        "target_mode": ctx.config.target_mode,
        "input_sql": ctx.sql,
        "chosen_path": chosen_path,
        "estimated_selectivity": ctx.selectivity,
        "estimated_filtered_rows": ctx.filtered_rows,
        "estimated_recall": recall,
        "expected_qps_class": qps,
        "required_indexes": [],
        "session_settings": [],
        "rewritten_sql": ctx.sql.strip().rstrip(";") + ";",
        "validation_sql": validation_sql(),
        "why_safe": "",
        "why_faster_than_default": "",
        "fallback_plan": "",
        "stats": ctx.stats,
        "assumptions": list(ctx.assumptions),
        "missing_stats": list(ctx.missing_stats),
    }


def recommended_indexes(ctx: RuleContext) -> list[dict[str, str]]:
    if ctx.config.index_budget == "no_new_indexes":
        return []
    shape = ctx.shape
    indexes = ctx.indexes
    out: list[dict[str, str]] = []
    if shape.where_clause and not (has_index(indexes, "using btree") or has_index(indexes, "using hash")):
        out.append({
            "type": "btree_or_partial",
            "ddl": "-- Create B-tree or partial index on high-frequency WHERE filter columns.",
        })
    if not has_index(indexes, "using hnsw") and not has_index(indexes, "using ivfflat"):
        vector_col = shape.vector_column or "embedding"
        ops = metric_ops(shape.metric_operator)
        table_ref = table_regclass(shape.schema, shape.table)
        out.append({
            "type": "hnsw",
            "ddl": f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{shape.table}_{vector_col}_hnsw ON {table_ref} USING hnsw ({qident(vector_col)} {ops}) WITH (m = 16, ef_construction = 128);",
        })
    return out


class PgStatsCollector:
    """Read-only catalog/stat collector. Uses estimates; avoids full COUNT(*)."""

    def __init__(self, dsn: str, connect_timeout: int = 10):
        if psycopg is None:
            raise RuntimeError("psycopg is not installed. Run: pip install -r requirements.txt")
        self.dsn = dsn
        self.connect_timeout = connect_timeout

    def collect(self, sql: str, default_schema: str = "public") -> dict[str, Any]:
        shape = parse_query_shape(sql, default_schema)
        reg = table_regclass(shape.schema, shape.table)
        out: dict[str, Any] = {"query_shape": asdict(shape), "stats_source": ["db_catalog", "explain_json"]}

        with psycopg.connect(self.dsn, connect_timeout=self.connect_timeout) as conn:
            conn.execute("SET LOCAL statement_timeout = '10s'")
            out["pgvector_version"] = self._scalar(conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            out["estimated_rows"] = self._scalar(conn, "SELECT reltuples::bigint FROM pg_class WHERE oid = %s::regclass", [reg])
            out["indexes"] = self._rows(conn, "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s", [shape.schema, shape.table])
            out["column_stats"] = self._rows(conn, "SELECT attname, n_distinct, most_common_vals::text, most_common_freqs::text, histogram_bounds::text FROM pg_stats WHERE schemaname = %s AND tablename = %s", [shape.schema, shape.table])
            if shape.vector_column:
                try:
                    out["vector_dimension"] = self._scalar(conn, f"SELECT vector_dims({qident(shape.vector_column)}) FROM {reg} WHERE {qident(shape.vector_column)} IS NOT NULL LIMIT 1")
                except Exception:
                    out["vector_dimension"] = None
            out["filtered_rows_estimate"] = self._explain_rows(conn, shape, reg)

        n = out.get("estimated_rows") or 0
        nf = out.get("filtered_rows_estimate")
        out["estimated_selectivity"] = (float(nf) / float(n)) if n and nf is not None else None
        return out

    @staticmethod
    def _scalar(conn, query: str, params: list[Any] | None = None) -> Any:
        row = conn.execute(query, params or []).fetchone()
        return row[0] if row else None

    @staticmethod
    def _rows(conn, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cur = conn.execute(query, params or [])
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    @staticmethod
    def _explain_rows(conn, shape: QueryShape, reg: str) -> int | None:
        predicate = shape.where_clause or "TRUE"
        try:
            raw = conn.execute(f"EXPLAIN (FORMAT JSON) SELECT 1 FROM {reg} WHERE {predicate}").fetchone()[0]
            plan = raw[0]["Plan"] if isinstance(raw, list) else json.loads(raw)[0]["Plan"]
            return int(plan.get("Plan Rows"))
        except Exception:
            return None


@register_rule
def rule_exact_prefilter_for_low_cardinality(ctx: RuleContext) -> Optional[dict[str, Any]]:
    """Exact pre-filter is safest when filtered rows are small or selectivity is extremely low."""
    n = ctx.estimated_rows
    nf = ctx.filtered_rows
    s = ctx.selectivity
    k = ctx.shape.limit_k
    exact_threshold = size_based_exact_threshold(n, ctx.config)
    scalar_index = has_index(ctx.indexes, "using btree") or has_index(ctx.indexes, "using hash")

    if nf is None:
        return None

    should_exact = nf <= exact_threshold
    should_exact = should_exact or (s is not None and s < 0.01 and scalar_index)
    should_exact = should_exact or (ctx.config.target_mode == "absolute_recall" and nf <= max(exact_threshold * 5, k * 500))
    if not should_exact:
        return None

    plan = base_plan(ctx, "exact_prefilter", 1.0, "medium" if nf <= exact_threshold else "low")
    plan["required_indexes"] = recommended_indexes(ctx)
    plan["session_settings"] = ["-- Optional: SET LOCAL enable_seqscan = off; only when planner misestimates scalar index path."]
    plan["why_safe"] = "先标量过滤，再对过滤后的向量集合做精确距离排序；召回等价 exact，不会发生 ANN 后过滤候选不足。"
    plan["why_faster_than_default"] = "过滤后行数较少或选择度极低时，比默认 ANN 先召回再过滤更稳定，也避免无效候选。"
    plan["fallback_plan"] = "如果 filtered_rows 变大或延迟过高，切换到 hnsw_filtered_iterative 并用 exact baseline 校验 overlap recall。"
    return plan


@register_rule
def rule_hnsw_filtered_iterative(ctx: RuleContext) -> Optional[dict[str, Any]]:
    """Default safe ANN path for filtered pgvector queries."""
    s = ctx.selectivity
    if s is None:
        s = 0.10
        ctx.assumptions.append("过滤选择度未知，保守假设 s=0.10。")
        ctx.missing_stats.append("estimated_selectivity")

    k = ctx.shape.limit_k
    required_candidates = math.ceil(k / max(float(s), 1e-6))

    if ctx.config.target_mode == "absolute_recall":
        ef = clamp(required_candidates * 4, 200, 5000)
        recall = 0.99
        qps = "low"
        iterative = "strict_order"
        max_scan = max(ef * 100, 100_000)
    elif ctx.config.target_mode == "recall_first":
        ef = clamp(required_candidates * 2, 100, 2000)
        recall = max(ctx.config.high_recall_target, 0.95)
        qps = "medium"
        iterative = "strict_order"
        max_scan = max(ef * 80, 50_000)
    elif ctx.config.target_mode == "balanced":
        ef = clamp(required_candidates * 1.5, 80, 1200)
        recall = 0.90
        qps = "medium-high"
        iterative = "strict_order"
        max_scan = max(ef * 60, 30_000)
    else:
        ef = clamp(required_candidates * 1.2, 64, 800)
        recall = max(ctx.config.min_acceptable_recall, 0.75)
        qps = "high"
        iterative = "relaxed_order" if s >= 0.20 else "strict_order"
        max_scan = max(ef * 50, 20_000)

    if recall < ctx.config.min_acceptable_recall:
        return None

    plan = base_plan(ctx, "hnsw_filtered_iterative", recall, qps)
    plan["required_indexes"] = recommended_indexes(ctx)
    plan["session_settings"] = [
        f"SET LOCAL hnsw.ef_search = {ef};",
        f"SET LOCAL hnsw.iterative_scan = {iterative};",
        f"SET LOCAL hnsw.max_scan_tuples = {max_scan};",
        "SET LOCAL hnsw.scan_mem_multiplier = 2;",
    ]

    base_sql = ctx.sql.strip().rstrip(";")
    if iterative == "relaxed_order":
        plan["rewritten_sql"] = (
            "WITH ann AS MATERIALIZED (\n"
            f"  {base_sql}\n"
            ")\n"
            f"SELECT * FROM ann ORDER BY {ctx.shape.order_by or '1'} LIMIT {k};"
        )
    else:
        plan["rewritten_sql"] = base_sql + ";"

    plan["why_safe"] = (
        f"按 K/selectivity 估算至少需要 {required_candidates} 个候选，并显式设置 "
        f"hnsw.ef_search={ef}, hnsw.iterative_scan={iterative}；避免默认 ef_search=40 在过滤后候选不足导致低召回。"
    )
    plan["why_faster_than_default"] = "相比 exact 大集合排序，HNSW 提供更高 QPS；相比 pgvector 默认参数，按过滤选择度放大候选并启用 iterative scan。"
    plan["fallback_plan"] = "如果实测 recall 未达标，先将 ef_search 和 max_scan_tuples 翻倍；仍不足则切 exact_prefilter 或 partial_hnsw。"
    return plan


# Example extension point:
# @register_rule
# def rule_my_new_index(ctx: RuleContext) -> Optional[dict[str, Any]]:
#     if not has_index(ctx.indexes, "using my_index"):
#         return None
#     plan = base_plan(ctx, "my_new_path", 0.97, "high")
#     plan["session_settings"] = ["SET LOCAL my_index.search_width = 200;"]
#     return plan


@register_prompt_fragment
def prompt_core_safety(cfg: OptimizerConfig) -> str:
    return f"""
Hard safety rules:
1. Never choose a plan with estimated recall < {cfg.min_acceptable_recall}.
2. For HNSW + WHERE, never rely on default hnsw.ef_search=40 unless post-filter candidates >= 2K.
3. For HNSW + WHERE, explicitly set hnsw.iterative_scan. Prefer strict_order for absolute_recall and recall_first.
4. If using relaxed_order, wrap the ANN query in a MATERIALIZED CTE and re-rank by true distance.
5. If filtered cardinality is small or selectivity is extremely low, prefer exact pre-filter.
"""


@register_prompt_fragment
def prompt_target_modes(cfg: OptimizerConfig) -> str:
    return f"""
Target modes:
- absolute_recall: maximize recall; exact or high-width ANN is acceptable.
- recall_first: target recall >= {cfg.high_recall_target}, but avoid very slow plans.
- balanced: balance recall and QPS; target recall >= 0.90 with good performance.
- performance_first: maximize QPS while keeping recall >= {cfg.min_acceptable_recall}.
"""


@register_prompt_fragment
def prompt_format_rules(cfg: OptimizerConfig) -> str:
    return """
PlanSpec format rules (CRITICAL — violations cause runtime errors):
1. `rewritten_sql` MUST be a single, directly-executable SELECT statement. Never embed SET commands in it.
2. `session_settings` MUST be an array of PostgreSQL SET LOCAL commands with full syntax: "SET LOCAL <name> = <value>;". Never use bare parameter names like "hnsw.ef_search" — those are NOT valid SQL.
3. `rewritten_sql` MUST preserve the original LIMIT value exactly. If the input SQL has LIMIT 100, your rewritten_sql MUST also use LIMIT 100. Do not change the limit.
4. When using CTEs (WITH clauses), always use table-qualified column names to avoid ambiguity, e.g. `my_table.image_vec`, not bare `image_vec`.
5. The rewritten_sql must return the same columns in the same order as the original query.
6. Every active session_setting must include "=" and a value, e.g. "SET LOCAL hnsw.iterative_scan = strict_order;".
7. Comment-only settings like "-- Optional: ..." are allowed but must NOT be the only setting — always include at least one active SET LOCAL.
8. IMPORTANT: pgvector does NOT have a `enable_hnswscan` GUC parameter. To force a sequential scan (exact pre-filter), use: "SET LOCAL enable_indexscan = off;" and "SET LOCAL enable_bitmapscan = off;". Valid pgvector GUCs are: hnsw.ef_search, hnsw.iterative_scan, hnsw.max_scan_tuples, hnsw.scan_mem_multiplier, ivfflat.probes, ivfflat.iterative_scan.
9. To force use of a specific index scan, use "SET LOCAL enable_seqscan = off;" and "SET LOCAL enable_bitmapscan = off;".
"""


def build_llm_messages(sql: str, stats: dict[str, Any], cfg: OptimizerConfig) -> list[dict[str, str]]:
    fragments = "\n".join(fn(cfg) for fn in PROMPT_FRAGMENTS)
    system = f"""
You are an expert PostgreSQL + pgvector optimizer for filtered vector similarity search.
Output one JSON PlanSpec only. Do not output Markdown.
{fragments}
Return exactly {cfg.candidate_count} candidate PlanSpecs, ranked best-first.
Return JSON in this shape: {{"candidates": [PlanSpec, ...]}}.

PlanSpec fields:
candidate_id, version, target_mode, input_sql, chosen_path, estimated_selectivity, estimated_filtered_rows,
estimated_recall, expected_qps_class, required_indexes, session_settings, rewritten_sql,
validation_sql, why_safe, why_faster_than_default, fallback_plan, stats, assumptions, missing_stats.
candidate_id must be "candidate_1", "candidate_2", ... in order.
""".strip()
    user = f"""
Optimize this pgvector filtered ANN query.

SQL:
```sql
{sql}
```

Target mode: {cfg.target_mode}
Runtime constraints:
- max_latency_ms: {cfg.max_latency_ms}
- index_budget: {cfg.index_budget}
- min_acceptable_recall: {cfg.min_acceptable_recall}
- high_recall_target: {cfg.high_recall_target}
- candidate_count: {cfg.candidate_count}

Database stats:
```json
{json.dumps(stats, ensure_ascii=False, indent=2, default=str)}
```
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_candidates(obj: Any) -> list[dict[str, Any]]:
    """Accept {"candidates": [...]}, a single PlanSpec dict, or a raw list."""
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        return obj["candidates"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("LLM output is not a PlanSpec or candidates array")


def _recover_json(raw: str) -> Any:
    """Attempt to recover a malformed/truncated JSON string from LLM output.

    Strategies (tried in order):
    1. Strip markdown ```json fences
    2. Find the last complete object/array boundary and truncate
    3. Find the last complete line and truncate
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strategy 1: extract from markdown code fences
    fence_patterns = [
        (r"```json\s*([\s\S]*?)\s*```", 1),
        (r"```\s*([\s\S]*?)\s*```", 1),
    ]
    for pattern, group in fence_patterns:
        m = re.search(pattern, text)
        if m:
            text = m.group(group).strip()
            break

    # Strategy 2: find last complete object or array
    for closer in ["]", "}"]:
        pos = text.rfind(closer)
        if pos >= 0:
            candidate = text[: pos + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Strategy 3: find last complete line that looks like JSON
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].rstrip(",")
        candidate = "\n".join(lines[:i]) + "\n" + stripped
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def local_candidate_plans(sql: str, stats: dict[str, Any], cfg: OptimizerConfig) -> list[dict[str, Any]]:
    """Generate several deterministic candidate plans without LLM."""
    modes: list[TargetMode] = ["recall_first", "balanced", "performance_first", "absolute_recall"]
    if cfg.target_mode in modes:
        modes.remove(cfg.target_mode)
    modes.insert(0, cfg.target_mode)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mode in modes:
        c = OptimizerConfig(**{**asdict(cfg), "target_mode": mode})
        plan = PgVectorLLMOptimizer(config=c).optimize_rule_based(sql, stats)
        key = json.dumps({
            "chosen_path": plan.get("chosen_path"),
            "session_settings": plan.get("session_settings"),
            "rewritten_sql": plan.get("rewritten_sql"),
        }, sort_keys=True)
        if key not in seen:
            seen.add(key)
            plan["candidate_id"] = f"candidate_{len(out) + 1}"
            out.append(plan)
        if len(out) >= cfg.candidate_count:
            break
    return out


def execute_sql_with_settings(conn, sql: str, settings: list[str] | None = None, fetch: bool = True):
    """Execute SQL with optional session settings. Returns fetched rows if fetch=True.

    Settings are written as SET (session-level, not SET LOCAL) so they persist
    across autocommit statements on the same connection.
    """
    for setting in settings or []:
        s = setting.strip()
        if not s or s.startswith("--"):
            continue
        if not (s.upper().startswith("SET ") or s.upper().startswith("SET LOCAL ")):
            continue
        if "=" not in s:
            continue
        # Convert SET LOCAL to SET so it works in autocommit mode
        s_fixed = s
        if s.upper().startswith("SET LOCAL "):
            s_fixed = "SET " + s[10:].strip()
        try:
            conn.execute(s_fixed)
        except Exception as e:
            print(f"  [WARNING] Failed to apply setting '{s_fixed}': {e}", flush=True)

    clean_sql = sql
    if clean_sql.upper().lstrip().startswith("SET "):
        select_pos = clean_sql.upper().find("SELECT")
        if select_pos != -1:
            clean_sql = clean_sql[select_pos:]

    cur = conn.execute(clean_sql)
    if fetch:
        return cur.fetchall()
    return None


def _result_ids(rows) -> set[int]:
    """Extract ID set from query result rows."""
    if rows is None:
        return set()
    ids: set[int] = set()
    for row in rows:
        if isinstance(row, (tuple, list)):
            ids.add(int(row[0]))
        else:
            ids.add(int(row))
    return ids


def compute_recall(result_ids: set[int], ground_truth_ids: set[int]) -> float:
    """Compute recall = |result ∩ GT| / |GT|."""
    if not ground_truth_ids:
        return 0.0
    return len(result_ids & ground_truth_ids) / len(ground_truth_ids)


def benchmark_sql(
    dsn: str,
    sql: str,
    settings: list[str] | None = None,
    warmup: int = 2,
    runs: int = 10,
) -> dict[str, Any]:
    """Run a query repeatedly and return latency/QPS + result IDs from the last run."""
    if psycopg is None:
        raise RuntimeError("psycopg is not installed. Run: pip install -r requirements.txt")
    latencies_ms: list[float] = []
    last_result_ids: set[int] = set()
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        conn.execute("SET statement_timeout = '60s'")
        for _ in range(max(0, warmup)):
            execute_sql_with_settings(conn, sql, settings)
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            rows = execute_sql_with_settings(conn, sql, settings)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            last_result_ids = _result_ids(rows)
    total_s = sum(latencies_ms) / 1000.0
    return {
        "runs": len(latencies_ms),
        "concurrency": 1,
        "latency_ms_avg": statistics.mean(latencies_ms),
        "latency_ms_p50": statistics.median(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "qps": len(latencies_ms) / total_s if total_s > 0 else None,
        "result_ids": last_result_ids,
    }


def _benchmark_worker(dsn: str, sql: str, settings: list[str] | None, times: int) -> list[float]:
    latencies_ms: list[float] = []
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        for _ in range(times):
            t0 = time.perf_counter()
            execute_sql_with_settings(conn, sql, settings)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def benchmark_sql_concurrent(
    dsn: str,
    sql: str,
    settings: list[str] | None = None,
    times: int = 1,
    concurrency: int = 50,
    warmup: int = 0,
) -> dict[str, Any]:
    """
    Benchmark in the same spirit as benchmark.py:
    - concurrency = number of concurrent workers
    - times = how many times each worker runs the query
    - total queries = concurrency * times
    """
    if psycopg is None:
        raise RuntimeError("psycopg is not installed. Run: pip install -r requirements.txt")

    # Optional single-connection warmup.
    if warmup > 0:
        benchmark_sql(dsn, sql, settings, warmup=0, runs=warmup)

    all_latencies: list[float] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(_benchmark_worker, dsn, sql, settings, max(1, times))
            for _ in range(max(1, concurrency))
        ]
        for fut in as_completed(futures):
            all_latencies.extend(fut.result())
    wall_s = time.perf_counter() - start

    if not all_latencies:
        raise RuntimeError("No benchmark result collected")

    sorted_lats = sorted(all_latencies)
    p95_idx = min(len(sorted_lats) - 1, int(math.ceil(len(sorted_lats) * 0.95)) - 1)
    return {
        "runs": len(all_latencies),
        "times_per_worker": times,
        "concurrency": concurrency,
        "wall_time_s": wall_s,
        "latency_ms_avg": statistics.mean(all_latencies),
        "latency_ms_p50": statistics.median(all_latencies),
        "latency_ms_p95": sorted_lats[p95_idx],
        "latency_ms_min": min(all_latencies),
        "latency_ms_max": max(all_latencies),
        "qps": len(all_latencies) / wall_s if wall_s > 0 else None,
    }


def format_plan_compact(plan: dict[str, Any] | None) -> str:
    """Return a one-line summary of a PlanSpec suitable for display."""
    if plan is None:
        return "N/A"
    recall = plan.get("estimated_recall", "?")
    if isinstance(recall, float):
        recall = f"{recall:.3f}"
    path = plan.get("chosen_path", "?")
    return f"recall={recall}  path={path}"


def print_results_compact(
    results: list[dict[str, Any]],
    query_name: str | None = None,
    sql: str | None = None,
) -> None:
    """Print benchmark + plan results in a compact human-readable format."""
    if query_name:
        print(f"\n{'='*80}")
        print(f"  {query_name}")
        print(f"{'='*80}")
    if sql:
        print(f"  SQL: {sql}")
        print()

    print(f"  {'':20s} {'latency':>10s}  {'recall':>8s}  plan")
    print(f"  {'':20s} {'':->10s}  {'':->8s}  ----")

    for entry in results:
        name = entry["name"]
        bench = entry.get("benchmark")
        plan = entry.get("plan")
        # Use measured recall from benchmark, fall back to plan's estimated recall
        measured_recall = entry.get("recall")
        if measured_recall is not None:
            recall_str = f"{measured_recall:.3f}"
        elif plan and isinstance(plan.get("estimated_recall"), (int, float)):
            recall_str = f"{plan['estimated_recall']:.3f}"
        else:
            recall_str = "N/A"

        path_str = plan.get("chosen_path", "(baseline)") if plan else "(baseline)"

        if bench:
            latency = f"{bench['latency_ms_avg']:.2f}ms"
            print(f"  {name:<20s} {latency:>10s}  {recall_str:>8s}  {path_str}")
        elif plan:
            print(f"  {name:<20s} {'N/A':>10s}  {recall_str:>8s}  {path_str}")

    print()


class PgVectorLLMOptimizer:
    def __init__(self, dsn: str | None = None, config: OptimizerConfig | None = None):
        self.dsn = dsn or os.getenv("PG_DSN")
        self.config = config or OptimizerConfig()

    def collect_stats(self, sql: str) -> dict[str, Any]:
        if not self.dsn:
            raise ValueError("Missing PostgreSQL DSN. Pass --dsn or set PG_DSN.")
        return PgStatsCollector(self.dsn).collect(sql, self.config.default_schema)

    def optimize_rule_based(self, sql: str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        shape = parse_query_shape(sql, self.config.default_schema)
        stats = stats or self.collect_stats(sql)
        ctx = RuleContext(sql=sql, config=self.config, shape=shape, stats=stats)

        if ctx.filtered_rows is None and ctx.estimated_rows and ctx.selectivity is not None:
            ctx.stats["filtered_rows_estimate"] = int(ctx.estimated_rows * ctx.selectivity)
            ctx.assumptions.append("filtered_rows_estimate 由 estimated_rows * estimated_selectivity 推导。")

        if ctx.selectivity is None:
            ctx.missing_stats.append("estimated_selectivity")
        if not shape.vector_column:
            ctx.missing_stats.append("vector_column")

        for rule in RULES:
            plan = rule(ctx)
            if plan is not None:
                plan["assumptions"] = list(dict.fromkeys(plan.get("assumptions", []) + ctx.assumptions))
                plan["missing_stats"] = list(dict.fromkeys(plan.get("missing_stats", []) + ctx.missing_stats))
                return plan

        raise RuntimeError("No optimizer rule produced a plan. Add a fallback rule with @register_rule.")

    def optimize_rule_based_candidates(self, sql: str, stats: dict[str, Any] | None = None, candidate_count: int | None = None) -> list[dict[str, Any]]:
        stats = stats or self.collect_stats(sql)
        cfg = OptimizerConfig(**{**asdict(self.config), "candidate_count": candidate_count or self.config.candidate_count})
        return local_candidate_plans(sql, stats, cfg)

    def optimize_with_llm(self, sql: str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        candidates = self.optimize_with_llm_candidates(sql, stats)
        return candidates[0]

    def optimize_with_llm_candidates(self, sql: str, stats: dict[str, Any] | None = None, candidate_count: int | None = None) -> list[dict[str, Any]]:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from e

        stats = stats or self.collect_stats(sql)
        cfg = OptimizerConfig(**{**asdict(self.config), "candidate_count": candidate_count or self.config.candidate_count})
        messages = build_llm_messages(sql, stats, cfg)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://llmapi.blsc.cn/v1")

        max_retries = 2
        for attempt in range(max_retries):
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )

            raw = resp.choices[0].message.content or "{}"
            try:
                candidates = normalize_candidates(json.loads(raw))
                for idx, c in enumerate(candidates):
                    c["candidate_id"] = f"candidate_{idx + 1}"
                return candidates[: cfg.candidate_count]
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"  [LLM] JSON parse error (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                    print(f"  [LLM] retrying ...", flush=True)
                    continue
                # Last attempt: try recovery strategies
                print(f"  [LLM] JSON parse error on final attempt: {e}", flush=True)
                print(f"  [LLM] attempting recovery of malformed JSON ...", flush=True)
                parsed = _recover_json(raw)
                if parsed is not None:
                    try:
                        candidates = normalize_candidates(parsed)
                        for idx, c in enumerate(candidates):
                            c["candidate_id"] = f"candidate_{idx + 1}"
                        print(f"  [LLM] recovered {len(candidates)} candidates from partial JSON", flush=True)
                        return candidates[: cfg.candidate_count]
                    except (ValueError, KeyError) as ve:
                        print(f"  [LLM] recovery parse succeeded but content invalid: {ve}", flush=True)
                raise RuntimeError(
                    f"LLM returned malformed JSON that could not be recovered. "
                    f"Raw response (first 500 chars): {raw[:500]}"
                ) from e

        # Should never reach here
        return []

    def benchmark_candidates(
        self,
        sql: str,
        candidates: list[dict[str, Any]],
        ground_truth_ids: set[int] | None = None,
        warmup: int = 2,
        runs: int = 10,
        times: int | None = None,
        concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.dsn:
            raise ValueError("Missing PostgreSQL DSN. Pass --dsn, config.pg, or PG_DSN.")
        gt = ground_truth_ids or set()
        results: list[dict[str, Any]] = []

        def _run_bench(sql_to_run: str, settings: list[str] | None) -> dict[str, Any]:
            if concurrency and concurrency > 1:
                return benchmark_sql_concurrent(self.dsn, sql_to_run, settings, warmup=warmup, times=times or runs, concurrency=concurrency)
            else:
                return benchmark_sql(self.dsn, sql_to_run, settings, warmup=warmup, runs=runs)

        def _run_once_for_ids(sql_to_run: str, settings: list[str] | None) -> set[int]:
            """Run the query once to get result IDs for recall computation."""
            if psycopg is None:
                return set()
            try:
                with psycopg.connect(self.dsn) as conn:
                    conn.autocommit = True
                    rows = execute_sql_with_settings(conn, sql_to_run, settings)
                    return _result_ids(rows)
            except Exception:
                return set()

        base = _run_bench(sql, [])
        base_recall = compute_recall(_run_once_for_ids(sql, []), gt) if gt else None
        results.append({"name": "baseline", "benchmark": base, "recall": base_recall})

        for i, plan in enumerate(candidates, 1):
            rewritten = plan.get("rewritten_sql") or sql
            settings = plan.get("session_settings") or []
            b = _run_bench(rewritten, settings)
            rec = compute_recall(_run_once_for_ids(rewritten, settings), gt) if gt else None
            results.append({"name": f"candidate_{i}", "benchmark": b, "recall": rec, "plan": plan})
        return results


def self_test() -> None:
    sql = "SELECT id FROM items WHERE category_id = 1 ORDER BY embedding <=> $1 LIMIT 20"
    stats = {
        "estimated_rows": 1_000_000,
        "filtered_rows_estimate": 80_000,
        "estimated_selectivity": 0.08,
        "indexes": [
            {"indexname": "idx_items_embedding_hnsw", "indexdef": "CREATE INDEX ON public.items USING hnsw (embedding vector_cosine_ops)"},
            {"indexname": "idx_items_category", "indexdef": "CREATE INDEX ON public.items USING btree (category_id)"},
        ],
    }
    opt = PgVectorLLMOptimizer(config=OptimizerConfig(target_mode="recall_first"))
    plan = opt.optimize_rule_based(sql, stats)
    assert plan["chosen_path"] == "hnsw_filtered_iterative"
    assert any("hnsw.iterative_scan = strict_order" in x for x in plan["session_settings"])
    assert plan["estimated_recall"] >= 0.95

    low_stats = dict(stats, filtered_rows_estimate=1000, estimated_selectivity=0.001)
    low_plan = opt.optimize_rule_based(sql, low_stats)
    assert low_plan["chosen_path"] == "exact_prefilter"
    assert low_plan["estimated_recall"] == 1.0

    perf = PgVectorLLMOptimizer(config=OptimizerConfig(target_mode="performance_first"))
    perf_stats = dict(stats, filtered_rows_estimate=600_000, estimated_selectivity=0.60)
    perf_plan = perf.optimize_rule_based(sql, perf_stats)
    assert "MATERIALIZED" in perf_plan["rewritten_sql"]
    print("self-test passed")


def _strip_stats_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields needed for execution, drop raw pg_stats blobs."""
    keep = {
        "candidate_id", "chosen_path", "estimated_recall", "estimated_selectivity",
        "estimated_filtered_rows", "expected_qps_class",
        "session_settings", "rewritten_sql", "validation_sql",
        "why_safe", "why_faster_than_default", "fallback_plan",
        "target_mode", "required_indexes", "assumptions", "missing_stats",
    }
    return {k: v for k, v in plan.items() if k in keep}


def _compact_candidate_file(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact candidate file suitable for the test phase."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        item: dict[str, Any] = {
            "query_name": entry["query_name"],
            "sql": entry["generated_sql"],
            "candidates": [_strip_stats_from_plan(p) for p in entry.get("candidates", [])],
        }
        out.append(item)
    return out


def _resolve_table_name(args: argparse.Namespace) -> str:
    """Resolve table_name from --table-name or --schema-config."""
    if args.table_name:
        return args.table_name
    if args.schema_config:
        schema_cfg = load_schema_config(args.schema_config, "pgvector")
        name = schema_cfg.get("table_name")
        if name:
            return name
    raise ValueError("--table-name or --schema-config is required")


def _load_queries(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """Load queries from YAML or return a single synthetic query from --sql."""
    if args.yaml_queries:
        dataset, queries = load_vecbench_queries(args.yaml_queries, args.dataset)
        if args.query_name:
            queries = [q for q in queries if q.get("name") == args.query_name]
            if not queries:
                raise SystemExit(f"No query named '{args.query_name}' found in {args.yaml_queries}")
        return dataset, queries
    elif args.sql:
        return "_sql_", [{"name": "query", "sql": args.sql}]
    else:
        raise SystemExit("Either --sql or --yaml-queries must be provided")


def _sql_for_query(q: dict[str, Any], table_name: str, distance_ops: str) -> str:
    """Get SQL for a query dict, supporting both VecBench and raw --sql formats."""
    if "sql" in q:
        return q["sql"]
    return yaml_query_to_sql(q, table_name, distance_ops)


# Default config path — same directory as this script
_default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="pgvector LLM optimizer — init (plan) / test (benchmark)")
    parser.add_argument("--self-test", action="store_true", help="Run built-in smoke tests")
    sub = parser.add_subparsers(dest="phase", help="Phase: init or test")

    # ---- shared args ----
    shared_parent = argparse.ArgumentParser(add_help=False)
    shared_parent.add_argument("--sql", default=None, help="Filtered pgvector SQL (single-query mode)")
    shared_parent.add_argument("--yaml-queries", default=None, help="VecBench query YAML file")
    shared_parent.add_argument("--dataset", default=None, help="Dataset key in YAML")
    shared_parent.add_argument("--query-name", default=None, help="Process only this query")
    shared_parent.add_argument("--table-name", default=None, help="Database table name")
    shared_parent.add_argument("--schema-config", default=None, help="VecBench schema.yaml")
    shared_parent.add_argument("--distance-ops", default="vector_cosine_ops",
                               choices=["vector_l2_ops", "vector_cosine_ops", "vector_ip_ops", "vector_l1_ops"])
    shared_parent.add_argument("--config", default=_default_config, help="JSON config path")
    shared_parent.add_argument("--dsn", default=None, help="PostgreSQL DSN")
    shared_parent.add_argument("--mode", default=None, choices=["absolute_recall", "recall_first", "performance_first", "balanced"])
    shared_parent.add_argument("--candidates", type=int, default=3, help="Number of candidates per query")
    shared_parent.add_argument("--stats-json", help="Use local stats JSON instead of connecting to DB")

    # ---- init ----
    p_init = sub.add_parser("init", parents=[shared_parent], help="Generate candidate plans, save to file")
    p_init.add_argument("--output", default="candidates.json", help="Output candidate file")
    p_init.add_argument("--llm", action="store_true", help="Use OpenAI LLM instead of rule-based planner")
    p_init.add_argument("--model", default="DeepSeek-V4-Flash", help="LLM model name")

    # ---- test ----
    p_test = sub.add_parser("test", parents=[shared_parent], help="Benchmark candidates from file")
    p_test.add_argument("--input", required=True, help="Candidate JSON file from init phase")
    p_test.add_argument("--benchmark", action="store_true", default=True, help="Run benchmark (default: True)")
    p_test.add_argument("--warmup", type=int, default=2, help="Benchmark warmup runs")
    p_test.add_argument("--runs", type=int, default=10, help="Benchmark measured runs")
    p_test.add_argument("--times", type=int, default=None, help="Runs per concurrent worker")
    p_test.add_argument("--ground-truth", default=None, help="VecBench ground truth JSON (e.g. data/SIFT/ground_truth/E2E_1M_128.json)")
    p_test.add_argument("--concurrency", type=int, default=None, help="Concurrent workers")

    args = parser.parse_args()

    # --self-test works without phase
    if args.self_test:
        self_test()
        return

    if not args.phase:
        parser.print_help()
        return

    app_cfg = load_app_config(args.config) if args.config and os.path.exists(args.config) else {}
    cfg = optimizer_config_from_app(app_cfg, mode=args.mode, candidates=args.candidates)
    if hasattr(args, "model") and args.model:
        cfg.model = args.model
    dsn = dsn_from_config(app_cfg, args.dsn)

    # ---- INIT PHASE ----
    if args.phase == "init":
        table_name = _resolve_table_name(args)
        dataset, queries = _load_queries(args)

        opt = PgVectorLLMOptimizer(dsn=dsn, config=cfg)

        all_entries: list[dict[str, Any]] = []
        for q in queries:
            qname = q.get("name", "unnamed")
            sql = _sql_for_query(q, table_name, args.distance_ops)

            stats = None
            if args.stats_json:
                with open(args.stats_json, "r", encoding="utf-8") as f:
                    stats = json.load(f)

            if args.llm:
                print(f"  [LLM] generating {cfg.candidate_count} candidates for {qname} ...", flush=True)
                candidates = opt.optimize_with_llm_candidates(sql, stats, cfg.candidate_count)
            else:
                print(f"  [rule] generating {cfg.candidate_count} candidates for {qname} ...", flush=True)
                candidates = opt.optimize_rule_based_candidates(sql, stats, cfg.candidate_count)

            entry = {
                "query_name": qname,
                "generated_sql": sql,
                "candidates": candidates,
            }
            all_entries.append(entry)

        compact = _compact_candidate_file(all_entries)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2, default=str)

        n_queries = len(compact)
        n_candidates = sum(len(e["candidates"]) for e in compact)
        print(f"\nSaved {n_queries} queries × ~{n_candidates // max(n_queries, 1)} candidates → {args.output}")
        return

    # ---- TEST PHASE ----
    if args.phase == "test":
        with open(args.input, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # Load ground truth if provided
        gt_map: dict[str, set[int]] = {}
        if args.ground_truth:
            with open(args.ground_truth, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            for item in gt_data:
                gt_map[item["name"]] = set(item["result"])

        opt = PgVectorLLMOptimizer(dsn=dsn, config=cfg)

        for entry in entries:
            qname = entry["query_name"]
            sql = entry["sql"]
            candidates = entry["candidates"]
            gt_ids = gt_map.get(qname, set())

            bench_results = opt.benchmark_candidates(
                sql,
                candidates,
                ground_truth_ids=gt_ids,
                warmup=args.warmup,
                runs=args.runs,
                times=args.times,
                concurrency=args.concurrency,
            )
            print_results_compact(bench_results, query_name=qname, sql=sql)

        return


if __name__ == "__main__":
    main()