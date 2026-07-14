# pgvector LLM Optimizer — Quick Start

文件位置：

```text
baseline/vecLLM/
├── pgvector_llm_optimizer.py   # 入口脚本
├── config.json                 # 连接 + 优化 + benchmark 配置
├── quick_start.md              # 本文件
└── requirements.txt            # （可选）依赖
```

功能：

- **init**：从 VecBench YAML 查询文件 或 裸 SQL 生成候选 PlanSpec（规则 / LLM），保存到 JSON
- **test**：加载候选文件，跑 benchmark，打印 compact 格式的 recall + latency
- 支持 4 种优化模式：`absolute_recall` / `recall_first` / `balanced` / `performance_first`
- 支持 pgvector HNSW + WHERE 场景：显式设置 `hnsw.iterative_scan` 和 `hnsw.ef_search`，防止过滤后候选不足导致召回崩塌

---

## 1. 安装

```bash
pip install psycopg pyyaml python-dotenv
# LLM 模式额外需要：
pip install openai
```

---

## 2. 配置 config.json

```json
{
  "pg": {
    "host": "localhost",
    "port": 9432,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
  },
  "optimizer": {
    "target_mode": "balanced",
    "candidate_count": 5,
    "min_acceptable_recall": 0.70,
    "high_recall_target": 0.95,
    "max_latency_ms": 200,
    "index_budget": "allow_new_indexes",
    "default_schema": "public",
    "model": "DeepSeek-V4-Flash"
  },
  "benchmark": {
    "warmup_runs": 2,
    "runs": 10,
    "times": 1,
    "concurrency": 50
  }
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `pg.dsn` | 完整连接串，优先级最高（与 host/port/database/user/password 二选一） |
| `optimizer.target_mode` | `absolute_recall` / `recall_first` / `balanced` / `performance_first` |
| `optimizer.candidate_count` | 每个 query 生成几个候选计划 |
| `optimizer.index_budget` | `no_new_indexes` / `allow_new_indexes` / `allow_partial_indexes` |
| `benchmark.times` | 每个并发 worker 跑几次（并发模式下） |
| `benchmark.concurrency` | 并发 worker 数，>1 则走并发 benchmark |

> **DSN 优先级**：`--dsn` CLI > `config.pg.dsn` > `config.pg.{host,port,database,user,password}` > `PG_DSN` 环境变量

---

## 3. 初始化：生成候选计划

### 3.1 从 VecBench YAML 查询文件（推荐）

```bash
cd /home/liujianzhong/VecBench

# 规则版 — 不调用 LLM，秒级完成
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --yaml-queries config/SIFT/E2E_queries_1M.yaml \
  --schema-config config/SIFT/schema.yaml \
  --table-name my_table \
  --distance-ops vector_l2_ops \
  --mode balanced \
  --candidates 10 \
  --output candidates_balanced.json \
  --query-name query1

# LLM 版 — 调用 LLM 生成候选
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --yaml-queries config/SIFT/E2E_queries_1M.yaml \
  --schema-config config/SIFT/schema.yaml \
  --table-name my_table \
  --distance-ops vector_l2_ops \
  --mode balanced \
  --candidates 10 \
  --output ./llm_candidates.json \
  --llm \
  --query-name query1
```

### 3.2 从裸 SQL（单条查询）

```bash
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --sql "SELECT id FROM items WHERE category_id = 1 ORDER BY embedding <=> '[...]' LIMIT 20" \
  --mode recall_first \
  --candidates 3 \
  --output single_candidates.json
```

### 3.3 使用离线 stats（不连 DB）

```bash
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --sql "..." \
  --stats-json path/to/stats.json \
  --output offline_candidates.json
```

### 3.4 输出文件结构

```json
[
  {
    "query_name": "query1",
    "sql": "SELECT id FROM my_table WHERE ... ORDER BY ... LIMIT 100",
    "candidates": [
      {
        "chosen_path": "hnsw_filtered_iterative",
        "estimated_recall": 0.95,
        "estimated_selectivity": 0.01,
        "session_settings": [
          "SET LOCAL hnsw.ef_search = 200;",
          "SET LOCAL hnsw.iterative_scan = strict_order;"
        ],
        "rewritten_sql": "SELECT id FROM my_table WHERE ... ORDER BY ... LIMIT 100;",
        "why_safe": "...",
        "why_faster_than_default": "..."
      }
    ]
  }
]
```

---

## 4. 测试：Benchmark 候选计划

```bash
# 单线程（默认 runs=10）
python baseline/vecLLM/pgvector_llm_optimizer.py test \
  --input candidates_balanced.json \
  --ground-truth data/SIFT/ground_truth/E2E_1M_128.json \
  --warmup 0 --runs 10

# 并发模式（对齐 benchmark.py 风格）
python baseline/vecLLM/pgvector_llm_optimizer.py test \
  --input candidates_balanced.json \
  --ground-truth data/SIFT/ground_truth/E2E_1M_128.json \
  --concurrency 50 --times 1 --warmup 2
```

运行流程：

1. 先跑原始 SQL（baseline），记录 latency + recall
2. 逐个跑每个候选 PlanSpec：
   - 执行 `session_settings`
   - 执行 `rewritten_sql`
   - 计算实测 recall（与 ground truth 交集 / GT）
3. 打印 compact 对比表

示例输出：

```text
================================================================================
  query1
================================================================================
  SQL: SELECT id FROM my_table WHERE equal = 27 ORDER BY ...
  
                      latency    recall  plan
                      ----------  --------  ----
  baseline             8.31ms     0.920  (baseline)
  candidate_1          3.83ms     0.987  hnsw_filtered_iterative
  candidate_2          3.22ms     0.964  hnsw_filtered_iterative
  candidate_3          2.95ms     0.931  hnsw_filtered_iterative
```

### 可选参数

| 参数 | 说明 |
|------|------|
| `--ground-truth` | VecBench GT JSON 路径，不传则不计算 recall |
| `--query-name` | 只测试指定 query |
| `--concurrency` | >1 则走并发 benchmark（`times * concurrency` 总请求） |
| `--times` | 每个并发 worker 的重复次数 |
| `--dsn` | 覆盖 config.json 中的连接串 |

---

## 5. 自测

```bash
python baseline/vecLLM/pgvector_llm_optimizer.py --self-test
```

测试两个内置 case：
- `recall_first` + 中等过滤率 → `hnsw_filtered_iterative`，recall ≥ 0.95
- 极低过滤率 → `exact_prefilter`，recall = 1.0

---

## 6. 一键 init + test（最常用）

```bash
# 规则生成 + benchmark（不需要 API key）
python baseline/vecLLM/pgvector_llm_optimizer.py init \
  --yaml-queries config/SIFT/E2E_queries_1M.yaml \
  --schema-config config/SIFT/schema.yaml \
  --table-name my_table \
  --distance-ops vector_l2_ops \
  --mode balanced \
  --candidates 5 \
  --output candidates.json

python baseline/vecLLM/pgvector_llm_optimizer.py test \
  --input candidates.json \
  --ground-truth data/SIFT/ground_truth/.json \
  --warmup 2 --runs 10
```

---

## 7. 扩展

### 新增规则

```python
@register_rule
def rule_partial_hnsw_for_tenant(ctx: RuleContext) -> Optional[dict[str, Any]]:
    if "tenant_id" not in (ctx.shape.where_clause or ""):
        return None

    plan = base_plan(ctx, "partial_index", 0.97, "high")
    plan["required_indexes"] = [
        {
            "type": "partial_hnsw",
            "ddl": "CREATE INDEX CONCURRENTLY ... WHERE tenant_id = ...;"
        }
    ]
    plan["session_settings"] = [
        "SET LOCAL hnsw.ef_search = 200;",
        "SET LOCAL hnsw.iterative_scan = strict_order;"
    ]
    return plan
```

### 新增 LLM 提示词片段

```python
@register_prompt_fragment
def prompt_company_rule(cfg: OptimizerConfig) -> str:
    return "For tenant_id filters, prefer partition pruning or partial HNSW before global HNSW."
```

---

## 8. 支持的优化模式

| 模式 | 策略 | 典型 ef_search | 目标 recall |
|------|------|---------------|-------------|
| `absolute_recall` | 最大化召回，可接受 exact 或大宽度 ANN | 200–5000 | ≥ 0.99 |
| `recall_first` | 召回优先，但避免极慢计划 | 100–2000 | ≥ 0.95 |
| `balanced` | 均衡召回与 QPS | 80–1200 | ≥ 0.90 |
| `performance_first` | QPS 优先，保证最低可接受召回 | 64–800 | ≥ 0.75 |