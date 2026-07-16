import argparse
import os

from databases.pgvector import PGVector
from databases.milvus import Milvus
from databases.qdrant import Qdrant
from databases.weaviate import Weaviate

from data.load_yfcc import load_dataset_with_scalars, perform_incremental_load, perform_update, perform_delete

from utils.data_synthesizer import adjust_dimension, adjust_scale, generate_incremental_data
from utils.query_generator import gen_queries_random
from utils.analyzer import Analyzer
from utils.workload_executor import (prepare_config, load_query_from_yaml,
                                     load_ground_truth, execute_save, execute)
from utils.concurrent import setup_database_c, execute_concurrent, execute_concurrent_hits
from utils.plot import save_sorted_results, plot_distribution


def _get_data_dimension(dataset):
    """Read vector dimension from the actual data file (not schema)."""
    import h5py
    data_dir = f"data/{dataset}"
    if not os.path.isdir(data_dir):
        return None
    # Known non-vector HDF5 keys (ground truth / meta)
    SKIP_KEYS = {'distances', 'neighbors', 'test', 'neighbours'}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(('.hdf5', '.h5')):
            with h5py.File(os.path.join(data_dir, fname), 'r') as f:
                candidates = [(k, f[k].shape) for k in f
                              if isinstance(f[k], h5py.Dataset)
                              and len(f[k].shape) == 2
                              and k.lower() not in SKIP_KEYS
                              and f[k].shape[1] > 100]  # dimension floor
                if candidates:
                    return max(candidates, key=lambda x: x[1][0])[1][1]
    # Try u8bin files (YFCC)
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.u8bin'):
            with open(os.path.join(data_dir, fname), 'rb') as f:
                f.read(4)  # skip nvecs
                return int.from_bytes(f.read(4), 'little')
    return None


def _format_scale(scale):
    """Format scale integer to human-readable string, e.g. 1000000 -> '1M'."""
    if scale >= 1_000_000 and scale % 1_000_000 == 0:
        return f"{scale // 1_000_000}M"
    elif scale >= 1_000 and scale % 1_000 == 0:
        return f"{scale // 1_000}K"
    return str(scale)


def get_query_path(dataset, scale):
    """Generate dataset-aware query YAML path: E2E_queries_{scale}.yaml (for saving)."""
    scale_str = _format_scale(scale)
    return f"config/{dataset}/E2E_queries_{scale_str}.yaml"


def get_gt_path(dataset, scale):
    """Generate dataset-aware ground-truth path: E2E_{scale}_{dim}.json"""
    dim = _get_data_dimension(dataset)
    scale_str = _format_scale(scale)
    if dim is not None:
        path = f"data/{dataset}/ground_truth/E2E_{scale_str}_{dim}.json"
        if os.path.exists(path):
            return path
    # Fallback: search for any ground-truth JSON file
    import glob
    candidates = sorted(glob.glob(f"data/{dataset}/ground_truth/E2E_*.json"))
    if candidates:
        return candidates[0]
    if dim is not None:
        return path  # return the intended path so the error message is clear
    raise ValueError(f"Cannot determine vector dimension for dataset {dataset}")


def get_test_query_path(dataset, scale):
    """Generate dataset-aware test query path: E2E_queries_{scale}.yaml (for loading)."""
    scale_str = _format_scale(scale)
    path = f"config/{dataset}/E2E_queries_{scale_str}.yaml"
    if os.path.exists(path):
        return path
    # Fallback: find any E2E_queries_*.yaml that exists
    import glob
    candidates = sorted(glob.glob(f"config/{dataset}/E2E_queries_*.yaml"))
    if candidates:
        return candidates[0]
    return path  # return the intended path so the error message is clear

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--case", help="The benchmark execution status , data_pre or init or modify_queries or test.", default='init',)
    parser.add_argument("--database", help="the vector database to benchmark", default='milvus',)
    parser.add_argument("--dataset", help="the choice of dataset ", default='YFCC',)
    parser.add_argument("--scale", type=int, default=10_000_000, help="Scale of the test data you want")
    parser.add_argument("--chunk_rows", type=int, default=1_000_000, help="Chunk size for data loading")
    parser.add_argument("--regen_incremental", action="store_true", help="Force regenerate incremental data")
    parser.add_argument("--algorithm", help="the algorithm to be tested", default='hnsw',)
    parser.add_argument("--times", help="the times every single query runs", type=int, default=1,)
    parser.add_argument("--concurrency", help="number of concurrent threads", type=int, default=50)
    parser.add_argument("--ratio", help="the ratio of data to CURD", type=float, default=0.2,)
    parser.add_argument("--in_ratio", help="the ratio of data to insert before creating index", type=float, default=0.2,)
    parser.add_argument("--up_ratio", help="the ratio of data to update", type=float, default=0.2,)
    parser.add_argument("--de_ratio", help="the ratio of data to delete", type=float, default=0.2,)
    parser.add_argument("--table_name", help="Override table/collection name (default: from schema config)", default=None,)
    args = parser.parse_args()
    return args
 
def setup_database(args, config):
    if args.database == 'pgvector':
        db = PGVector(config, args.database)
    elif args.database == 'milvus':
        db = Milvus(config, args.database)
    elif args.database == 'qdrant':
        db = Qdrant(config, args.database)
    elif args.database == 'weaviate':
        db = Weaviate(config, args.database)
    else:
        raise ValueError("Only support 'pgvector'、'milvus'、'qdrant'、'weaviate' now.")
    db.connect()
    print(f"------{db.db_type} connect successfully.")
    return db

def main(args):

    if args.case == 'data_pre':
        print("================debug1=================")

        if args.dataset == 'YFCC':
            print("================debug2=================")
            adjust_dimension(f"data/{args.dataset}", vec_file="base.10M.u8bin", new_dim=1920)
            adjust_scale(f"data/{args.dataset}", args, target_scale=100_000_000)
            generate_incremental_data(f"data/{args.dataset}", args)
        elif args.dataset == 'SIFT':
            print("================debug3=================")
            # 为 SIFT 生成标量文件
            import h5py
            import numpy as np
            data_dir = f"data/{args.dataset}"
            hdf5_path = os.path.join(data_dir, "sift-128-euclidean.hdf5")
            bin_path = os.path.join(data_dir, "sift_scalar.bin")
            
            if not os.path.exists(hdf5_path):
                print(f"错误: 找不到 {hdf5_path}，请先下载 SIFT 数据集")
            else:
                with h5py.File(hdf5_path, 'r') as f:
                    n_total = f['train'].shape[0]
                # 使用固定种子生成可复现的标量值 (0-99)
                rng = np.random.default_rng(42)
                scalar_data = rng.integers(0, 100, size=(n_total,), dtype=np.uint16)
                with open(bin_path, 'wb') as f:
                    f.write(np.int32(n_total).tobytes())
                    f.write(np.int32(1).tobytes())
                    scalar_data.tofile(f)
                print(f"已为 SIFT 生成标量文件: {bin_path} (n={n_total})")
        else:
            print(f"数据集 {args.dataset} 暂不支持 data_pre，请直接使用 init 和 test")
        
    elif args.case == 'init':
        db_config,index_config,schema_config = prepare_config(args)
        db = setup_database(args, db_config)

        # Override table_name from CLI if provided (before create_table reads schema)
        if args.table_name:
            schema_config[args.database]['table_name'] = args.table_name

        '''construct table or schema'''
        if (args.database == 'qdrant'):
            db.create_table(args.database, schema=schema_config, index = index_config)
        else:
            db.create_table(args.database, schema=schema_config)
        print(f"----{db.db_type} create table successfully.")

        loader_fn = load_dataset_with_scalars(f"./data/{args.dataset}", args, as_list=(db.db_type != 'milvus'))
        for df_chunk, _, _ in loader_fn():
            df_chunk = db.process_data(df_chunk, args.database, schema_config)
            db.insert_data(df_chunk, args.database, schema_config)

        # Auto-generate queries from the actual data so scalar values,
        # queries, and ground truth are always consistent.
        query_path = get_query_path(args.dataset, args.scale)
        print(f"Generating queries from actual data → {query_path} ...")
        outfile = gen_queries_random(loader_fn, args, n_queries=100, outfile=query_path)
        queries = load_query_from_yaml(args.dataset, outfile)

        dim = _get_data_dimension(args.dataset)
        scale_str = _format_scale(args.scale)
        gt_path = f"data/{args.dataset}/ground_truth/E2E_{scale_str}_{dim}.json"
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        print(f"Computing ground truth → {gt_path} ...")
        execute_save(queries, loader_fn, outpath=gt_path, save=True)

        # '''Initialization Phase'''
        print("P1 : Initialization phase is doing.")
        if args.algorithm and args.algorithm != 'flat':
            db.create_index(index_config[db.db_type], args)
            print(f"----{db.db_type} create index successfully.")
            if not (args.database == 'qdrant' and db.hnswp):
                db.create_scalar_index(index_config[db.db_type], args)
        else:
            print(f"----{db.db_type} skipping index creation (algorithm={args.algorithm or 'flat'})")
        print("P1 : Initialization phase is finished.")

        print("init finished.")

    elif args.case == 'modify_queries':
        db_config,index_config,schema_config = prepare_config(args)
        db = setup_database(args, db_config)  
        print("Loading original queries ...")
        orig_queries = load_query_from_yaml(args.dataset, f"config/{args.dataset}/E2E_queries_1M_locak_k.yaml")
        print("Preparing loader ...")
        loader_fn = load_dataset_with_scalars(f"./data/{args.dataset}", args, as_list=(db.db_type != 'milvus'))

        print("Running Analyzer ...")
        analyzer = Analyzer(
            loader_fn=loader_fn,
            queries=orig_queries,
            top_k=500,
            compute_filter_rates=True,
            compute_relevances=True,
            tags_denominator="occurrences",
        )

        default_filter_rate = 0.1
        default_relevance_rate = 0.1
        per_query_filter = {
            # "q001": 0.03,
            # "q015": 0.005,
        }
        per_query_relevance = {}
  
        print("Modifying queries ...")
        mode = 'relevance'
        analyzer.run(
            mode=mode,
            default_filter_rate=default_filter_rate,
            default_relevance_rate=default_relevance_rate,
            per_query_filter=per_query_filter,
            per_query_relevance=per_query_relevance,
        )

        out_yaml = f"config/{args.dataset}/E2E_queries_{mode}_modified.yaml"
        analyzer.save(out_yaml)
        print(f"Finished! Modified queries saved to: {out_yaml}")


    elif args.case == 'test':
        db_config,index_config,schema_config = prepare_config(args)
        db = setup_database(args, db_config)

        # Set correct table_name from schema (create_table normally does this, but test skips it)
        table_schema = schema_config.get(args.database, {})
        schema_table_name = table_schema.get("table_name", "my_table")
        db.table_name = args.table_name if args.table_name else schema_table_name
        effective_table_name = db.table_name

        # '''Initialization Phase'''
        # print("P1 : Initialization phase is doing.")
        # db.create_index(index_config[db.db_type], args)
        # print(f"----{db.db_type} create index successfully.")
        # if not (args.database == 'qdrant' and db.hnswp):
        #     db.create_scalar_index(index_config[db.db_type], args)
        # print("P1 : Initialization phase is finished.")

        '''Query Execution Phase'''
        print("P2 : Query execution phase is doing.")
        queries = load_query_from_yaml(args.dataset, get_test_query_path(args.dataset, args.scale))
        ground_truth = load_ground_truth(get_gt_path(args.dataset, args.scale))
        # detailed_results, overall_results = execute(db, queries, ground_truth, index_config["search_params"], args)
        # print(overall_results)
        # # save_sorted_results(detailed_results, prefix=f"{db.db_type}")
        # # plot_distribution(detailed_results, prefix=f"{db.db_type}")
        # # print("P2 : Query execution phase is finished.")

        '''Concurrent Phase'''
        print("P3 : Concurrent phase is doing.")
        db_factory = setup_database_c(args, db_config, table_name=effective_table_name)
        results = execute_concurrent(db_factory, queries, ground_truth, index_config["search_params"], args)

        # 对计算使用索引的比例
        # execute_concurrent_hits(db, queries, ground_truth, index_config["search_params"], args, db_config)

        # print(results)  # 输出特别多！
        print("P3 : Concurrent phase is finished.")

        '''Incremental Load Phase'''
        # print("P4 : Incremental load phase is doing.")
        # perform_incremental_load(
        #     db=db,
        #     base_data_dir=f"./data/{args.dataset}",
        #     schema=schema_config
        # )
        # print("P4 : Incremental load phase is finished.")

        '''Update Phase'''
        # print("P5 : Update phase is doing.")
        # perform_update(db=db, incremental_dir=f"./data/{args.dataset}/incremental_data", schema=schema_config, delta=1)
        # print("P5 : Update phase is finished.")

        '''Delete Phase'''
        # print("P6 : Delete phase is doing.")
        # perform_delete(db, incremental_dir=f"./data/{args.dataset}/incremental_data")
        # print("P6 : Delete phase is finished.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)