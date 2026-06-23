import yaml
import pandas as pd

def gen_queries_random(data, args, n_queries=1000, outfile="config/YFCC/E2E_queries.yaml"):
    loader = data
    all_chunks = []

    for df_chunk, _, _ in loader():
        all_chunks.append(df_chunk)

    all_data = pd.concat(all_chunks, ignore_index=True)
    n_sample = min(n_queries, len(all_data))
    sampled = all_data.sample(n=n_sample, random_state=42)

    queries = []
    qid = 1
    for _, row in sampled.iterrows():
        ref_id = row["id"]
        equal_val = row["equal"]
        range_val = row["range"]
        tags_list = row["tags"]

        # query1: equal
        queries.append({
            "name": f"query{qid}",
            "vector_field": "image_vec",
            "reference_vector_name": int(ref_id),
            "scalar_filters": [
                {
                    "field": "equal",
                    "operator": "==",
                    "value": int(equal_val),
                    "logic": "and"
                }
            ],
            "limit": 100
        })
        qid += 1

    query_dict = {f"{args.dataset}": queries}
    with open(outfile, "w") as f:
        yaml.dump(query_dict, f, sort_keys=False)
    return outfile