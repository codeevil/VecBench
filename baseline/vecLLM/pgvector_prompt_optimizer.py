"""
pgvector_prompt_optimizer.py

Prompt-driven pgvector filtered vector query optimizer.
All optimization knowledge lives in composable prompt string variables.
Minimal code — the optimizer is a prompt engine that assembles the right
instructions for the target mode, then delegates plan generation to rules
(rule-based mode) or an LLM (LLM mode).

Architecture:
  1. PROMPT LIBRARY  — all prompts as named string constants
  2. PROMPT COMPOSER — assemble prompts by mode/safety/format
  3. RULE ENGINE     — thin wrappers that compute params + fill prompt templates
  4. INFRASTRUCTURE  — DB stats, benchmarking, CLI (the only "real" code)

Usage:
  # rule-based init
  python baseline/vecLLM/pgvector_prompt_optimizer.py init \
    --yaml-queries config/SIFT/E2E_queries_1M.yaml \
    --schema-config config/SIFT/schema.yaml \
    --mode balanced --candidates 5 --output ./candidates.json

  # LLM init
  python baseline/vecLLM/pgvector_prompt_optimizer.py init --llm \
    --yaml-queries config/SIFT/100.E2E_queries_1M.yaml \
    --schema-config config/SIFT/schema.yaml \
    --mode recall_first --candidates 5

  # test
  python baseline/vecLLM/pgvector_prompt_optimizer.py test \
    --input candidates.json --ground-truth data/SIFT/ground_truth/E2E_1M_128.json

  # self-test
  python baseline/vecLLM/pgvector_prompt_optimizer.py --self-test
"""

# ============================================================================
# PROMPT LIBRARY
# All optimization knowledge lives here as composable string constants.
# Each prompt is self-contained and can be toggled on/off by the composer.
# ============================================================================

# ---- System role ----------------------------------------------------------

PROMPT_SYSTEM_ROLE = """\
You are an expert PostgreSQL + pgvector optimizer specializing in filtered \
vector similarity search (ANN + WHERE clauses). You analyze database catalog \
statistics and produce safe, performant PlanSpecs."""

# ---- Safety rules (database-agnostic) -------------------------------------

PROMPT_SAFETY_CORE = """\
## Core Safety Rules
1. Never choose a plan whose estimated recall falls below the minimum \
acceptable threshold for the target mode.
2. When filtered cardinality is small or selectivity is extremely low \
(< 1%), prefer exact pre-filter — it is safer and often faster.
3. Always recommend a fallback plan in case the primary plan underperforms."""

PROMPT_SAFETY_HNSW = """\
## HNSW + WHERE Safety (CRITICAL)
1. Never rely on default hnsw.ef_search=40 for filtered queries — the \
post-filter candidate pool will almost certainly be too small.
2. Always explicitly set hnsw.iterative_scan. Use `strict_order` for \
absolute_recall and recall_first modes.
3. When selectivity is unknown, assume s=0.10 and size ef_search accordingly.
4. If using relaxed_order, wrap the ANN query in a MATERIALIZED CTE and \
re-rank by true distance.
5. Set hnsw.max_scan_tuples to bound the worst-case scan cost."""

PROMPT_SAFETY_IVFFLAT = """\
## IVFFlat + WHERE Safety
1. Explicitly set ivfflat.probes — the default (1) is almost never sufficient \
for filtered queries.
2. Enable ivfflat.iterative_scan for filtered workloads to avoid early-exit \
with insufficient candidates.
3. For strict recall targets, pair high probes with a MATERIALIZED CTE re-rank."""

# ---- Strategy descriptions ------------------------------------------------

PROMPT_STRATEGY_EXACT_PREFILTER = """\
## Strategy: exact_prefilter
**When to use:**
- Filtered row count <= size-based threshold (small tables: 50K, medium: 20K, large: 10K)
- Selectivity < 1% AND a scalar index (btree/hash) exists on the filter column
- Target mode is absolute_recall and filtered rows are manageable

**What it does:**
1. Execute the WHERE clause first against scalar indexes (or seq scan if no index)
2. Compute exact distances on the filtered subset
3. Sort by distance and return top-K

**Trade-offs:**
- Recall: 1.0 (exact — no approximation)
- QPS: medium to low, degrades as filtered_rows grows
- Best for: low-cardinality filters, strict recall requirements"""

PROMPT_STRATEGY_HNSW_ITERATIVE = """\
## Strategy: hnsw_filtered_iterative
**When to use:**
- Primary strategy for most filtered HNSW queries
- Filtered rows exceed the exact threshold
- Selectivity is moderate to high (> 1%)

**What it does:**
1. Set hnsw.ef_search = K / selectivity × multiplier (mode-dependent)
2. Enable hnsw.iterative_scan so pgvector keeps searching until K results \
survive the filter
3. Bound with hnsw.max_scan_tuples to prevent runaway scans

**Parameter formulas (per target mode):**
| Mode              | ef_search        | iterative_scan  | recall target |
| absolute_recall   | candidates × 4   | strict_order    | ≥ 0.99        |
| recall_first      | candidates × 2   | strict_order    | ≥ 0.95        |
| balanced          | candidates × 1.5 | strict_order    | ≥ 0.90        |
| performance_first | candidates × 1.2 | relaxed_order*  | ≥ 0.75        |

*relaxed_order when selectivity ≥ 20%, else strict_order

**Trade-offs:**
- Recall: mode-dependent, 0.75–0.99
- QPS: medium to high
- Best for: most real-world filtered ANN workloads"""

PROMPT_STRATEGY_RELAXED_CTE = """\
## Strategy: relaxed_order with MATERIALIZED CTE re-rank
**When to use:**
- Performance-first mode with selectivity ≥ 20%
- Want maximum QPS while maintaining acceptable recall

**What it does:**
1. Use relaxed_order for fast approximate candidate collection
2. Wrap in MATERIALIZED CTE to force materialization
3. Re-rank candidates by true distance in the outer query

**SQL pattern:**
```sql
WITH ann AS MATERIALIZED (
  SELECT * FROM table WHERE ... ORDER BY vec <-> query LIMIT K
)
SELECT * FROM ann ORDER BY vec <-> query LIMIT K;
```"""

# ---- Target mode descriptions ---------------------------------------------

PROMPT_TARGET_MODES = """\
## Target Modes
- **absolute_recall**: Maximize recall above all else. Exact search or \
high-width ANN is acceptable. Target recall ≥ 0.99.
- **recall_first**: Prioritize recall (≥ 0.95) but avoid pathologically \
slow plans. Controlled-width ANN is preferred.
- **balanced**: Balance recall (≥ 0.90) and QPS equally. Good default.
- **performance_first**: Maximize QPS while keeping recall ≥ 0.75. \
Aggressive ANN parameters are acceptable."""

# ---- Format & output constraints ------------------------------------------

PROMPT_FORMAT_RULES = """\
## PlanSpec Format Rules (violations cause runtime errors)

### rewritten_sql
- MUST be a single, directly-executable SELECT statement
- Never embed SET commands in SQL — use session_settings for those
- MUST preserve the original LIMIT value exactly
- When using CTEs, use table-qualified column names to avoid ambiguity
- Must return the same columns in the same order as the original query

### session_settings
- MUST be an array of valid PostgreSQL SET LOCAL commands
- Full syntax required: "SET LOCAL <name> = <value>;"
- Never use bare parameter names — those are NOT valid SQL

### Valid pgvector GUCs
| GUC                        | Purpose                              |
| hnsw.ef_search              | HNSW candidate pool size             |
| hnsw.iterative_scan         | strict_order / relaxed_order         |
| hnsw.max_scan_tuples        | Hard limit on tuples scanned         |
| hnsw.scan_mem_multiplier    | Memory budget multiplier             |
| ivfflat.probes              | Number of IVF lists to probe         |
| ivfflat.iterative_scan      | strict_order / relaxed_order         |

### Forcing scan methods
- To force seq scan (exact pre-filter): SET LOCAL enable_indexscan = off; \
SET LOCAL enable_bitmapscan = off;
- To force index scan: SET LOCAL enable_seqscan = off; \
SET LOCAL enable_bitmapscan = off;

### Other
- Comment-only settings like "-- Optional: ..." are allowed but at least \
one active SET LOCAL must always be present.
- pgvector does NOT have an `enable_hnswscan` GUC parameter."""

PROMPT_OUTPUT_SCHEMA = """\
## Output Format
Return exactly {candidate_count} candidate PlanSpecs ranked best-first in:

```json
{{"candidates": [PlanSpec, ...]}}
```

### PlanSpec fields
- candidate_id: "candidate_1", "candidate_2", ... in rank order
- version: "1.0"
- target_mode: the target mode used
- input_sql: the original query SQL
- chosen_path: strategy name (exact_prefilter | hnsw_filtered_iterative | ...)
- estimated_selectivity: float or null
- estimated_filtered_rows: int or null
- estimated_recall: float (0.0–1.0)
- expected_qps_class: "low" | "medium" | "medium-high" | "high"
- required_indexes: [{{"type": "...", "ddl": "..."}}]
- session_settings: ["SET LOCAL ...;", ...]
- rewritten_sql: the optimized SQL string
- validation_sql: SQL to validate the plan (exact baseline comparison)
- why_safe: explanation of why this plan preserves recall (in Chinese)
- why_faster_than_default: explanation of performance improvement (in Chinese)
- fallback_plan: what to do if this plan underperforms (in Chinese)
- stats: {{}} (summary stats used)
- assumptions: [str, ...]
- missing_stats: [str, ...]"""

# ---- Plan explanation templates (filled by rule engine) -------------------

PROMPT_EXPLAIN_EXACT_WHY_SAFE = """\
先标量过滤，再对过滤后的向量集合做精确距离排序；召回等价 exact，\
不会发生 ANN 后过滤候选不足。"""

PROMPT_EXPLAIN_EXACT_WHY_FASTER = """\
过滤后行数较少或选择度极低时，比默认 ANN 先召回再过滤更稳定，\
也避免无效候选。"""

PROMPT_EXPLAIN_EXACT_FALLBACK = """\
如果 filtered_rows 变大或延迟过高，切换到 hnsw_filtered_iterative \
并用 exact baseline 校验 overlap recall。"""

PROMPT_EXPLAIN_HNSW_WHY_SAFE = """\
按 K/selectivity 估算至少需要 {required_candidates} 个候选，并显式设置 \
hnsw.ef_search={ef}, hnsw.iterative_scan={iterative}；\
避免默认 ef_search=40 在过滤后候选不足导致低召回。"""

PROMPT_EXPLAIN_HNSW_WHY_FASTER = """\
相比 exact 大集合排序，HNSW 提供更高 QPS；\
相比 pgvector 默认参数，按过滤选择度放大候选并启用 iterative scan。"""

PROMPT_EXPLAIN_HNSW_FALLBACK = """\
如果实测 recall 未达标，先将 ef_search 和 max_scan_tuples 翻倍；\
仍不足则切 exact_prefilter 或 partial_hnsw。"""

# ---- User message template ------------------------------------------------

PROMPT_USER_TEMPLATE = """\
Optimize this pgvector filtered ANN query.

SQL:
```sql
{sql}
```

Target mode: {target_mode}
Runtime constraints:
- max_latency_ms: {max_latency_ms}
- index_budget: {index_budget}
- min_acceptable_recall: {min_acceptable_recall}
- high_recall_target: {high_recall_target}
- candidate_count: {candidate_count}

Database stats:
```json
{stats_json}
```"""

# ---- Validation SQL template ----------------------------------------------

PROMPT_VALIDATION_SQL = """\
BEGIN;
SET LOCAL enable_indexscan = off;
SET LOCAL enable_bitmapscan = off;
-- Run exact baseline top-K, compare overlap with optimized top-K.
COMMIT;"""

# ---- DDL templates --------------------------------------------------------

PROMPT_DDL_BTREE_HINT = "CREATE INDEX CONCURRENTLY IF NOT EXISTS ON {table} USING btree (<filter columns>);"

PROMPT_DDL_HNSW_TEMPLATE = """\
CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} \
ON {table_ref} USING hnsw ({vector_col} {ops}) \
WITH (m = 16, ef_construction = 128);"""

# ---- Conservative defaults for missing stats ------------------------------

PROMPT_ASSUMPTION_UNKNOWN_SELECTIVITY = "过滤选择度未知，保守假设 s=0.10。"
PROMPT_ASSUMPTION_DERIVED_FILTERED = "filtered_rows_estimate 由 estimated_rows × estimated_selectivity 推导。"

PROMPT_MISSING_SELECTIVITY = "estimated_selectivity"
PROMPT_MISSING_VECTOR_COL = "vector_column"



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

# ============================================================================
# PROMPT COMPOSER
# Assembles the right prompt fragments for a given configuration.
# ============================================================================

def _indent(text: str, prefix: str = "") -> str:
    """Normalize and optionally prefix each non-empty line of a prompt fragment."""
    return text.strip()


def compose_system_prompt(config: "OptimizerConfig") -> str:
    """Compose the complete system prompt from registered fragments.

    Fragments are ordered: role → target modes → strategies → safety → format → output.
    """
    fragments: list[str] = []

    # 1. Role
    fragments.append(PROMPT_SYSTEM_ROLE)

    # 2. Target modes
    fragments.append(PROMPT_TARGET_MODES)

    # 3. Strategy catalog
    fragments.append(PROMPT_STRATEGY_EXACT_PREFILTER)
    fragments.append(PROMPT_STRATEGY_HNSW_ITERATIVE)
    if config.target_mode == "performance_first":
        fragments.append(PROMPT_STRATEGY_RELAXED_CTE)

    # 4. Safety rules
    fragments.append(PROMPT_SAFETY_CORE)
    fragments.append(PROMPT_SAFETY_HNSW)
    fragments.append(PROMPT_SAFETY_IVFFLAT)

    # 5. Format constraints
    fragments.append(PROMPT_FORMAT_RULES)

    # 6. Output schema
    fragments.append(PROMPT_OUTPUT_SCHEMA.format(candidate_count=config.candidate_count))

    return "\n\n".join(fragments)


def compose_user_prompt(sql: str, stats: dict[str, Any], config: "OptimizerConfig") -> str:
    """Fill the user message template with query context."""
    return PROMPT_USER_TEMPLATE.format(
        sql=sql,
        target_mode=config.target_mode,
        max_latency_ms=config.max_latency_ms,
        index_budget=config.index_budget,
        min_acceptable_recall=config.min_acceptable_recall,
        high_recall_target=config.high_recall_target,
        candidate_count=config.candidate_count,
        stats_json=json.dumps(stats, ensure_ascii=False, indent=2, default=str),
    )


def build_llm_messages(sql: str, stats: dict[str, Any], config: "OptimizerConfig") -> list[dict[str, str]]:
    """Build the final message list for the LLM chat API."""
    return [
        {"role": "system", "content": compose_system_prompt(config)},
        {"role": "user", "content": compose_user_prompt(sql, stats, config)},
    ]


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

TargetMode = Literal["absolute_recall", "recall_first", "performance_first", "balanced"]


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


# ============================================================================
# SQL UTILITIES
# ============================================================================

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


DISTANCE_OPS_TO_OPERATOR: dict[str, str] = {
    "vector_l2_ops": "<->",
    "vector_cosine_ops": "<=>",
    "vector_ip_ops": "<#>",
    "vector_l1_ops": "<+>",
}


def size_based_exact_threshold(n: int, cfg: OptimizerConfig) -> int:
    if n <= 100_000:
        return min(n or cfg.exact_scan_threshold_small, cfg.exact_scan_threshold_small)
    if n <= 1_000_000:
        return cfg.exact_scan_threshold_small
    if n <= 100_000_000:
        return cfg.exact_scan_threshold_medium
    return cfg.exact_scan_threshold_large


# ============================================================================
# RULE ENGINE
# Each rule is a thin function: compute params → fill prompt templates → return PlanSpec.
# All explanatory text comes from the PROMPT_* constants above.
# ============================================================================

def _base_plan(ctx: RuleContext, chosen_path: str, recall: float | None, qps: str) -> dict[str, Any]:
    """Create a PlanSpec skeleton. Text fields come from prompt constants."""
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
        "validation_sql": PROMPT_VALIDATION_SQL,
        "why_safe": "",
        "why_faster_than_default": "",
        "fallback_plan": "",
        "stats": ctx.stats,
        "assumptions": list(ctx.assumptions),
        "missing_stats": list(ctx.missing_stats),
    }


def _recommended_indexes(ctx: RuleContext) -> list[dict[str, str]]:
    if ctx.config.index_budget == "no_new_indexes":
        return []
    shape = ctx.shape
    indexes = ctx.indexes
    out: list[dict[str, str]] = []
    if shape.where_clause and not (has_index(indexes, "using btree") or has_index(indexes, "using hash")):
        out.append({
            "type": "btree_or_partial",
            "ddl": PROMPT_DDL_BTREE_HINT.format(table=table_regclass(shape.schema, shape.table)),
        })
    if not has_index(indexes, "using hnsw") and not has_index(indexes, "using ivfflat"):
        vector_col = shape.vector_column or "embedding"
        ops = metric_ops(shape.metric_operator)
        index_name = f"idx_{shape.table}_{vector_col}_hnsw"
        out.append({
            "type": "hnsw",
            "ddl": PROMPT_DDL_HNSW_TEMPLATE.format(
                index_name=index_name,
                table_ref=table_regclass(shape.schema, shape.table),
                vector_col=qident(vector_col),
                ops=ops,
            ),
        })
    return out


def rule_exact_prefilter(ctx: RuleContext) -> dict[str, Any] | None:
    """Exact pre-filter for low cardinality / extreme selectivity."""
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

    plan = _base_plan(ctx, "exact_prefilter", 1.0, "medium" if nf <= exact_threshold else "low")
    plan["required_indexes"] = _recommended_indexes(ctx)
    plan["session_settings"] = [
        "-- force seq scan for exact pre-filter; remove if scalar index path is better",
        "SET LOCAL enable_indexscan = off;",
        "SET LOCAL enable_bitmapscan = off;",
    ]
    plan["why_safe"] = PROMPT_EXPLAIN_EXACT_WHY_SAFE
    plan["why_faster_than_default"] = PROMPT_EXPLAIN_EXACT_WHY_FASTER
    plan["fallback_plan"] = PROMPT_EXPLAIN_EXACT_FALLBACK
    return plan


def rule_hnsw_filtered_iterative(ctx: RuleContext) -> dict[str, Any] | None:
    """Default safe ANN path for filtered pgvector queries."""
    s = ctx.selectivity
    if s is None:
        s = 0.10
        ctx.assumptions.append(PROMPT_ASSUMPTION_UNKNOWN_SELECTIVITY)
        ctx.missing_stats.append(PROMPT_MISSING_SELECTIVITY)

    k = ctx.shape.limit_k
    required_candidates = math.ceil(k / max(float(s), 1e-6))

    # ---- mode → (ef, recall, qps, iterative, max_scan) -----------------
    if ctx.config.target_mode == "absolute_recall":
        ef = clamp(required_candidates * 4, 200, 5000)
        recall, qps, iterative = 0.99, "low", "strict_order"
        max_scan = max(ef * 100, 100_000)
    elif ctx.config.target_mode == "recall_first":
        ef = clamp(required_candidates * 2, 100, 2000)
        recall, qps, iterative = max(ctx.config.high_recall_target, 0.95), "medium", "strict_order"
        max_scan = max(ef * 80, 50_000)
    elif ctx.config.target_mode == "balanced":
        ef = clamp(required_candidates * 1.5, 80, 1200)
        recall, qps, iterative = 0.90, "medium-high", "strict_order"
        max_scan = max(ef * 60, 30_000)
    else:  # performance_first
        ef = clamp(required_candidates * 1.2, 64, 800)
        recall, qps = max(ctx.config.min_acceptable_recall, 0.75), "high"
        iterative = "relaxed_order" if s >= 0.20 else "strict_order"
        max_scan = max(ef * 50, 20_000)

    if recall < ctx.config.min_acceptable_recall:
        return None

    plan = _base_plan(ctx, "hnsw_filtered_iterative", recall, qps)
    plan["required_indexes"] = _recommended_indexes(ctx)
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

    plan["why_safe"] = PROMPT_EXPLAIN_HNSW_WHY_SAFE.format(
        required_candidates=required_candidates, ef=ef, iterative=iterative,
    )
    plan["why_faster_than_default"] = PROMPT_EXPLAIN_HNSW_WHY_FASTER
    plan["fallback_plan"] = PROMPT_EXPLAIN_HNSW_FALLBACK
    return plan


# Rule registry — add new rules here to extend the optimizer
RULES: list[Callable[[RuleContext], dict[str, Any] | None]] = [
    rule_exact_prefilter,
    rule_hnsw_filtered_iterative,
]


# ============================================================================
# INFRASTRUCTURE — Stats collection, benchmarking, YAML loading
# ============================================================================

class PgStatsCollector:
    """Read-only catalog/stat collector using PostgreSQL estimates."""

    def __init__(self, dsn: str, connect_timeout: int = 10):
        if psycopg is None:
            raise RuntimeError("psycopg is not installed. Run: pip install psycopg")
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


# ---- Benchmarking ---------------------------------------------------------

def execute_sql_with_settings(conn, sql: str, settings: list[str] | None = None, fetch: bool = True):
    """Execute SQL with optional session settings."""
    for setting in settings or []:
        s = setting.strip()
        if not s or s.startswith("--"):
            continue
        if not (s.upper().startswith("SET ") or s.upper().startswith("SET LOCAL ")):
            continue
        if "=" not in s:
            continue
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
    if not ground_truth_ids:
        return 0.0
    return len(result_ids & ground_truth_ids) / len(ground_truth_ids)


def benchmark_sql(dsn: str, sql: str, settings: list[str] | None = None, warmup: int = 2, runs: int = 10) -> dict[str, Any]:
    if psycopg is None:
        raise RuntimeError("psycopg is not installed.")
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
    dsn: str, sql: str, settings: list[str] | None = None,
    times: int = 1, concurrency: int = 50, warmup: int = 0,
) -> dict[str, Any]:
    if psycopg is None:
        raise RuntimeError("psycopg is not installed.")
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


# ---- YAML / config loading ------------------------------------------------

def yaml_query_to_sql(query: dict[str, Any], table_name: str, distance_ops: str = "vector_cosine_ops") -> str:
    """Convert a VecBench-style query dict into a pgvector SQL string."""
    vector_field = query["vector_field"]
    reference_vector_name = query["reference_vector_name"]
    scalar_filters = query.get("scalar_filters", [])
    limit = query["limit"]
    distance_op = DISTANCE_OPS_TO_OPERATOR.get(distance_ops, "<=>")

    scalar_conditions: list[tuple[str, str]] = []
    for f in scalar_filters:
        field, operator, value = f["field"], f["operator"], f["value"]
        logic = f.get("logic", "and")

        cond_map = {
            "==": f"{field} = {value}",
            "<": f"{field} < '{value}'",
            "<=": f"{field} <= '{value}'",
            ">": f"{field} > '{value}'",
            ">=": f"{field} >= '{value}'",
            "like": f"{field} LIKE '%{value}%'",
            "contains": f"'{value}' = ANY({field})",
        }
        condition = cond_map.get(operator)
        if condition:
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
    return " ".join(sql.strip().split())


def load_vecbench_queries(yaml_path: str, dataset: str | None = None) -> tuple[str, list[dict[str, Any]]]:
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
    if yaml is None:
        raise RuntimeError("PyYAML is not installed.")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if database not in data:
        raise KeyError(f"Database '{database}' not found in schema config. Available: {list(data.keys())}")
    return data[database]


def load_app_config(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dsn_from_config(app_cfg: dict[str, Any], cli_dsn: str | None = None) -> str | None:
    if cli_dsn:
        return cli_dsn
    pg = app_cfg.get("pg", {}) if isinstance(app_cfg, dict) else {}
    if pg.get("dsn"):
        return pg["dsn"]
    if pg.get("host") and pg.get("database") and pg.get("user"):
        host, port, db = pg.get("host", "localhost"), pg.get("port", 5432), pg["database"]
        user, password = pg["user"], pg.get("password", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return os.getenv("PG_DSN")


def optimizer_config_from_app(app_cfg: dict[str, Any], mode: str | None = None, candidates: int | None = None) -> OptimizerConfig:
    raw = dict(app_cfg.get("optimizer", {}) or {})
    if mode:
        raw["target_mode"] = mode
    if candidates is not None:
        raw["candidate_count"] = candidates
    allowed = set(OptimizerConfig.__dataclass_fields__.keys())
    return OptimizerConfig(**{k: v for k, v in raw.items() if k in allowed})


# ---- LLM JSON recovery ----------------------------------------------------

def _recover_json(raw: str) -> Any:
    """Attempt to recover malformed/truncated JSON from LLM output."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    # Strategy 1: extract from markdown code fences
    for pattern, group in [(r"```json\s*([\s\S]*?)\s*```", 1), (r"```\s*([\s\S]*?)\s*```", 1)]:
        m = re.search(pattern, text)
        if m:
            text = m.group(group).strip()
            break

    # Strategy 2: find last complete object or array
    for closer in ["]", "}"]:
        pos = text.rfind(closer)
        if pos >= 0:
            try:
                return json.loads(text[: pos + 1])
            except json.JSONDecodeError:
                continue

    # Strategy 3: truncate at last complete JSON-like line
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        candidate = "\n".join(lines[:i]) + "\n" + lines[i].rstrip(",")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def normalize_candidates(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        return obj["candidates"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("LLM output is not a PlanSpec or candidates array")


# ---- Candidate generation -------------------------------------------------

def local_candidate_plans(sql: str, stats: dict[str, Any], cfg: OptimizerConfig) -> list[dict[str, Any]]:
    """Generate deterministic candidates by running rules in different target modes."""
    modes: list[TargetMode] = ["recall_first", "balanced", "performance_first", "absolute_recall"]
    if cfg.target_mode in modes:
        modes.remove(cfg.target_mode)
    modes.insert(0, cfg.target_mode)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mode in modes:
        c = OptimizerConfig(**{**asdict(cfg), "target_mode": mode})
        plan = PgVectorPromptOptimizer(config=c).optimize_rule_based(sql, stats)
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


# ---- Display --------------------------------------------------------------

def format_plan_compact(plan: dict[str, Any] | None) -> str:
    if plan is None:
        return "N/A"
    recall = plan.get("estimated_recall", "?")
    if isinstance(recall, float):
        recall = f"{recall:.3f}"
    return f"recall={recall}  path={plan.get('chosen_path', '?')}"


def print_results_compact(results: list[dict[str, Any]], query_name: str | None = None, sql: str | None = None) -> None:
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
        measured_recall = entry.get("recall")
        if measured_recall is not None:
            recall_str = f"{measured_recall:.3f}"
        elif plan and isinstance(plan.get("estimated_recall"), (int, float)):
            recall_str = f"{plan['estimated_recall']:.3f}"
        else:
            recall_str = "N/A"

        path_str = plan.get("chosen_path", "(baseline)") if plan else "(baseline)"
        latency = f"{bench['latency_ms_avg']:.2f}ms" if bench else "N/A"
        print(f"  {name:<20s} {latency:>10s}  {recall_str:>8s}  {path_str}")

    print()


def _strip_stats_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "candidate_id", "chosen_path", "estimated_recall", "estimated_selectivity",
        "estimated_filtered_rows", "expected_qps_class",
        "session_settings", "rewritten_sql", "validation_sql",
        "why_safe", "why_faster_than_default", "fallback_plan",
        "target_mode", "required_indexes", "assumptions", "missing_stats",
    }
    return {k: v for k, v in plan.items() if k in keep}


def _compact_candidate_file(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        out.append({
            "query_name": entry["query_name"],
            "sql": entry["generated_sql"],
            "candidates": [_strip_stats_from_plan(p) for p in entry.get("candidates", [])],
        })
    return out


# ============================================================================
# MAIN OPTIMIZER CLASS
# Thin orchestrator — delegates to rule engine or LLM.
# ============================================================================

class PgVectorPromptOptimizer:
    """Prompt-driven pgvector query optimizer.

    Two paths:
      - rule_based:  runs registered RULES, fills prompt templates for explanations
      - llm:         composes system+user prompts, calls OpenAI-compatible API
    """

    def __init__(self, dsn: str | None = None, config: OptimizerConfig | None = None):
        self.dsn = dsn or os.getenv("PG_DSN")
        self.config = config or OptimizerConfig()

    # -- Stats --------------------------------------------------------------

    def collect_stats(self, sql: str) -> dict[str, Any]:
        if not self.dsn:
            raise ValueError("Missing PostgreSQL DSN. Pass --dsn or set PG_DSN.")
        return PgStatsCollector(self.dsn).collect(sql, self.config.default_schema)

    # -- Rule-based optimization --------------------------------------------

    def optimize_rule_based(self, sql: str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run registered rules; return the first matching PlanSpec."""
        shape = parse_query_shape(sql, self.config.default_schema)
        stats = stats or self.collect_stats(sql)
        ctx = RuleContext(sql=sql, config=self.config, shape=shape, stats=stats)

        # Fill in missing estimates
        if ctx.filtered_rows is None and ctx.estimated_rows and ctx.selectivity is not None:
            ctx.stats["filtered_rows_estimate"] = int(ctx.estimated_rows * ctx.selectivity)
            ctx.assumptions.append(PROMPT_ASSUMPTION_DERIVED_FILTERED)

        if ctx.selectivity is None:
            ctx.missing_stats.append(PROMPT_MISSING_SELECTIVITY)
        if not shape.vector_column:
            ctx.missing_stats.append(PROMPT_MISSING_VECTOR_COL)

        for rule in RULES:
            plan = rule(ctx)
            if plan is not None:
                plan["assumptions"] = list(dict.fromkeys(plan.get("assumptions", []) + ctx.assumptions))
                plan["missing_stats"] = list(dict.fromkeys(plan.get("missing_stats", []) + ctx.missing_stats))
                return plan

        raise RuntimeError("No optimizer rule produced a plan. Add a fallback rule to RULES.")

    def optimize_rule_based_candidates(
        self, sql: str, stats: dict[str, Any] | None = None, candidate_count: int | None = None,
    ) -> list[dict[str, Any]]:
        stats = stats or self.collect_stats(sql)
        cfg = OptimizerConfig(**{**asdict(self.config), "candidate_count": candidate_count or self.config.candidate_count})
        return local_candidate_plans(sql, stats, cfg)

    # -- LLM optimization ---------------------------------------------------

    def optimize_with_llm(self, sql: str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        candidates = self.optimize_with_llm_candidates(sql, stats)
        return candidates[0]

    def optimize_with_llm_candidates(
        self, sql: str, stats: dict[str, Any] | None = None, candidate_count: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is not installed.") from e

        stats = stats or self.collect_stats(sql)
        cfg = OptimizerConfig(**{**asdict(self.config), "candidate_count": candidate_count or self.config.candidate_count})

        # ---- The key prompt composition happens here ----
        system_prompt = compose_system_prompt(cfg)
        user_prompt = compose_user_prompt(sql, stats, cfg)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://llmapi.blsc.cn/v1")

        for attempt in range(2):
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
                if attempt < 1:
                    print(f"  [LLM] JSON parse error (attempt {attempt + 1}/2): {e}", flush=True)
                    continue
                print(f"  [LLM] attempting recovery of malformed JSON ...", flush=True)
                parsed = _recover_json(raw)
                if parsed is not None:
                    try:
                        candidates = normalize_candidates(parsed)
                        for idx, c in enumerate(candidates):
                            c["candidate_id"] = f"candidate_{idx + 1}"
                        print(f"  [LLM] recovered {len(candidates)} candidates", flush=True)
                        return candidates[: cfg.candidate_count]
                    except (ValueError, KeyError) as ve:
                        print(f"  [LLM] recovery parse succeeded but content invalid: {ve}", flush=True)
                raise RuntimeError(
                    f"LLM returned malformed JSON. Raw (first 500 chars): {raw[:500]}"
                ) from e

        return []

    # -- Benchmarking -------------------------------------------------------

    def benchmark_candidates(
        self, sql: str, candidates: list[dict[str, Any]],
        ground_truth_ids: set[int] | None = None,
        warmup: int = 2, runs: int = 10,
        times: int | None = None, concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.dsn:
            raise ValueError("Missing PostgreSQL DSN.")
        gt = ground_truth_ids or set()
        results: list[dict[str, Any]] = []

        def _run_bench(sql_to_run: str, settings: list[str] | None) -> dict[str, Any]:
            if concurrency and concurrency > 1:
                return benchmark_sql_concurrent(self.dsn, sql_to_run, settings, warmup=warmup, times=times or runs, concurrency=concurrency)
            else:
                return benchmark_sql(self.dsn, sql_to_run, settings, warmup=warmup, runs=runs)

        def _run_once_for_ids(sql_to_run: str, settings: list[str] | None) -> set[int]:
            if psycopg is None:
                return set()
            try:
                with psycopg.connect(self.dsn) as conn:
                    conn.autocommit = True
                    return _result_ids(execute_sql_with_settings(conn, sql_to_run, settings))
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


# ============================================================================
# SELF-TEST
# ============================================================================

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
    opt = PgVectorPromptOptimizer(config=OptimizerConfig(target_mode="recall_first"))
    plan = opt.optimize_rule_based(sql, stats)
    assert plan["chosen_path"] == "hnsw_filtered_iterative", f"expected hnsw_filtered_iterative, got {plan['chosen_path']}"
    assert any("hnsw.iterative_scan = strict_order" in x for x in plan["session_settings"])
    assert plan["estimated_recall"] >= 0.95
    # Check that prompt-driven explanations are populated
    assert plan["why_safe"], "why_safe should be populated from prompt template"
    assert plan["why_faster_than_default"], "why_faster_than_default should be populated from prompt template"

    low_stats = dict(stats, filtered_rows_estimate=1000, estimated_selectivity=0.001)
    low_plan = opt.optimize_rule_based(sql, low_stats)
    assert low_plan["chosen_path"] == "exact_prefilter"
    assert low_plan["estimated_recall"] == 1.0
    assert low_plan["why_safe"] == PROMPT_EXPLAIN_EXACT_WHY_SAFE

    perf = PgVectorPromptOptimizer(config=OptimizerConfig(target_mode="performance_first"))
    perf_stats = dict(stats, filtered_rows_estimate=600_000, estimated_selectivity=0.60)
    perf_plan = perf.optimize_rule_based(sql, perf_stats)
    assert "MATERIALIZED" in perf_plan["rewritten_sql"]

    # Verify prompt composition works
    sys_prompt = compose_system_prompt(OptimizerConfig(target_mode="balanced", candidate_count=3))
    assert "balanced" in sys_prompt.lower() or "Balanced" in sys_prompt
    assert "candidate_count" in sys_prompt or "3" in sys_prompt

    print("self-test passed")


# ============================================================================
# CLI
# ============================================================================

_default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _resolve_table_name(args: argparse.Namespace) -> str:
    if args.table_name:
        return args.table_name
    if args.schema_config:
        schema_cfg = load_schema_config(args.schema_config, "pgvector")
        name = schema_cfg.get("table_name")
        if name:
            return name
    raise ValueError("--table-name or --schema-config is required")


def _load_queries(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
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
    if "sql" in q:
        return q["sql"]
    return yaml_query_to_sql(q, table_name, distance_ops)


def main() -> None:
    parser = argparse.ArgumentParser(description="pgvector prompt optimizer — init (plan) / test (benchmark)")
    parser.add_argument("--self-test", action="store_true", help="Run built-in smoke tests")
    sub = parser.add_subparsers(dest="phase", help="Phase: init or test")

    shared_parent = argparse.ArgumentParser(add_help=False)
    shared_parent.add_argument("--sql", default=None)
    shared_parent.add_argument("--yaml-queries", default=None)
    shared_parent.add_argument("--dataset", default=None)
    shared_parent.add_argument("--query-name", default=None)
    shared_parent.add_argument("--table-name", default=None)
    shared_parent.add_argument("--schema-config", default=None)
    shared_parent.add_argument("--distance-ops", default="vector_cosine_ops",
                               choices=["vector_l2_ops", "vector_cosine_ops", "vector_ip_ops", "vector_l1_ops"])
    shared_parent.add_argument("--config", default=_default_config)
    shared_parent.add_argument("--dsn", default=None)
    shared_parent.add_argument("--mode", default=None, choices=["absolute_recall", "recall_first", "performance_first", "balanced"])
    shared_parent.add_argument("--candidates", type=int, default=3)
    shared_parent.add_argument("--stats-json", help="Use local stats JSON instead of connecting to DB")

    p_init = sub.add_parser("init", parents=[shared_parent])
    p_init.add_argument("--output", default="candidates.json")
    p_init.add_argument("--llm", action="store_true")
    p_init.add_argument("--model", default="DeepSeek-V4-Flash")

    p_test = sub.add_parser("test", parents=[shared_parent])
    p_test.add_argument("--input", required=True)
    p_test.add_argument("--benchmark", action="store_true", default=True)
    p_test.add_argument("--warmup", type=int, default=2)
    p_test.add_argument("--runs", type=int, default=10)
    p_test.add_argument("--times", type=int, default=None)
    p_test.add_argument("--ground-truth", default=None)
    p_test.add_argument("--concurrency", type=int, default=None)

    args = parser.parse_args()

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

    if args.phase == "init":
        table_name = _resolve_table_name(args)
        dataset, queries = _load_queries(args)
        opt = PgVectorPromptOptimizer(dsn=dsn, config=cfg)

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

            all_entries.append({"query_name": qname, "generated_sql": sql, "candidates": candidates})

        compact = _compact_candidate_file(all_entries)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2, default=str)

        n_queries = len(compact)
        n_candidates = sum(len(e["candidates"]) for e in compact)
        print(f"\nSaved {n_queries} queries × ~{n_candidates // max(n_queries, 1)} candidates → {args.output}")
        return

    if args.phase == "test":
        with open(args.input, "r", encoding="utf-8") as f:
            entries = json.load(f)

        gt_map: dict[str, set[int]] = {}
        if args.ground_truth:
            with open(args.ground_truth, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            for item in gt_data:
                gt_map[item["name"]] = set(item["result"])

        opt = PgVectorPromptOptimizer(dsn=dsn, config=cfg)

        for entry in entries:
            bench_results = opt.benchmark_candidates(
                entry["sql"], entry["candidates"],
                ground_truth_ids=gt_map.get(entry["query_name"], set()),
                warmup=args.warmup, runs=args.runs,
                times=args.times, concurrency=args.concurrency,
            )
            print_results_compact(bench_results, query_name=entry["query_name"], sql=entry["sql"])
        return


if __name__ == "__main__":
    main()
