#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milvus 本地数据集性能测试脚本
支持数据集: sift1m, bigann1m, bigann10m, bigann100m
连接到本地 Milvus: localhost:19531
用法: 
    python test_milvus.py.py --name sift1m
    python test_milvus.py.py --name bigann1m --gt 0
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from pymilvus import (
    MilvusClient, DataType, CollectionSchema, FieldSchema,
    connections, Collection, utility
)

# ==================== 自定义读取函数 ====================
def fvecs_read(filename, dim=None):
    """读取 .fvecs 文件，返回 numpy array (float32)"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            d = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            vec = np.frombuffer(f.read(4 * d), dtype=np.float32)
            if dim and len(vec) != dim:
                print(f"警告: 向量维度 {len(vec)} 与期望 {dim} 不符")
            vectors.append(vec)
    return np.array(vectors)

def u8bin_read(filename, dim_bytes=128, limit=None, to_float=False):
    """读取 .u8bin 文件，返回 bytes 列表或 float32 数组"""
    vectors = []
    with open(filename, 'rb') as f:
        if limit is None:
            data = f.read()
            total = len(data) // dim_bytes
            for i in range(total):
                vectors.append(data[i*dim_bytes:(i+1)*dim_bytes])
        else:
            for _ in range(limit):
                vec = f.read(dim_bytes)
                if not vec or len(vec) != dim_bytes:
                    break
                vectors.append(vec)
    
    if to_float:
        import numpy as np
        vectors = np.array([np.frombuffer(v, dtype=np.uint8).astype(np.float32) for v in vectors])
    
    return vectors

def ivecs_read(filename, k=10):
    """读取 .ivecs 文件，返回 ground truth (n_query x k)"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            d = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            if d != k:
                print(f"警告: GT 维度 {d} 与期望 {k} 不符")
            vec = np.frombuffer(f.read(4 * d), dtype=np.int32)
            vectors.append(vec)
    return np.array(vectors)

def csv_gt_read(filename, k=10):
    """读取 gt_attr*_knn{k}.csv 文件，返回 ground truth (n_query x k)"""
    gt = []
    with open(filename, 'r') as f:
        for line in f:
            ids = [int(x) for x in line.strip().split(',')]
            gt.append(np.array(ids[:k], dtype=np.int32))
    return np.array(gt)

def compute_recall(results, groundtruth, top_k=10):
    """计算召回率"""
    total_recall = 0
    for i, (res, gt) in enumerate(zip(results, groundtruth)):
        if len(res) == 0:
            continue
        result_ids = [r['id'] for r in res[:top_k]]
        result_set = set(result_ids)
        gt_set = set(gt[:top_k])
        recall = len(result_set & gt_set) / min(top_k, len(gt_set))
        total_recall += recall
    return total_recall / len(results)

def read_attributes(csv_path, id_col='id'):
    """读取属性 CSV，返回 id -> {attr} 的字典（可选）"""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    # 假设第一列是 id
    if id_col in df.columns:
        return df.set_index(id_col).to_dict(orient='index')
    return {}

# ==================== 数据集配置 ====================
DATASETS = {
    'sift1m': {
        'base_file': 'sift_base.fvecs',
        'query_file': 'sift_query.fvecs',
        'gt_file': 'gt_attr3_knn10.csv',
        'attr_file': 'sift_attributes.csv',
        'dim': 128,
        'vector_type': 'float',
        'metric_type': 'L2',
        'index_type': 'HNSW',
        'index_nlist': 128,
        'index_m': 16,
        'index_ef_construction': 200,
        'search_ef': 64,
        'description': 'SIFT1M 浮点向量'
    },
    'bigann1m': {
        'base_file': 'base_1m_float.fvecs',
        'query_file': 'query_float.fvecs',
        'gt_file': 'gt_attr3_knn10.csv',
        'attr_file': 'bigann1M_attributes.csv',
        'dim': 128,
        'vector_type': 'float',
        'metric_type': 'L2',
        'index_type': 'HNSW',
        'index_m': 32,
        'index_ef_construction': 200,
        'search_ef': 128,
        'description': 'BIGANN 1M 浮点向量'
    },
    'bigann10m': {
        'base_file': 'base_10m_float.fvecs',
        'query_file': 'query_float.fvecs',
        'gt_file': 'gt_attr3_knn10.csv',
        'attr_file': 'bigann10M_attributes.csv',
        'dim': 128,
        'vector_type': 'float',
        'metric_type': 'L2',
        'index_type': 'HNSW',
        'index_m': 32,
        'index_ef_construction': 200,
        'search_ef': 64,
        'description': 'BIGANN 10M 浮点向量'
    },
    'bigann100m': {
        'base_file': 'base.1B.u8bin.crop_nb_100000000',
        'query_file': 'query.public.10K.u8bin',
        'gt_file': 'gt_attr3_knn10.csv',
        'attr_file': 'bigann100M_attributes.csv',
        'dim': 128,
        'vector_type': 'binary',
        'metric_type': 'HAMMING',
        'index_type': 'HNSW',
        'index_m': 32,
        'index_ef_construction': 200,
        'search_ef': 64,
        'description': 'BIGANN 100M 二进制向量'
    }
}


# ==================== Milvus 连接 ====================
def create_client(host='localhost', port='19531'):
    return MilvusClient(uri=f"http://{host}:{port}")

def create_collection(client, collection_name, cfg):
    """根据配置创建 Collection"""
    dim = cfg['dim']
    if cfg['vector_type'] == 'float':
        vector_field = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim)
    else:
        # BINARY_VECTOR 的 dim 是比特数，字节数 * 8
        binary_dim = dim * 8
        vector_field = FieldSchema(name="vec", dtype=DataType.BINARY_VECTOR, dim=binary_dim)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        vector_field,
        FieldSchema(name="attr", dtype=DataType.INT32)
    ]
    schema = CollectionSchema(fields, description=cfg['description'])
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type=cfg['metric_type']
    )
    print(f"✅ 创建集合 {collection_name}，维度 {dim}，类型 {cfg['vector_type']}")

def create_index(client, collection_name, cfg, index_name='vector_index'):
    """创建索引"""
    index_params = client.prepare_index_params()
    index_type = cfg.get('index_type', 'HNSW')
    if index_type == 'HNSW':
        params = {
            "M": cfg.get('index_m', 16),
            "efConstruction": cfg.get('index_ef_construction', 200)
        }
        if 'index_gamma' in cfg:
            params['gamma'] = cfg['index_gamma']
        if 'index_m_beta' in cfg:
            params['m_beta'] = cfg['index_m_beta']
        index_params.add_index(
            field_name="vec",
            index_type=index_type,
            metric_type=cfg['metric_type'],
            params=params,
            index_name=index_name
        )
    elif index_type in ['IVF_FLAT', 'BIN_IVF_FLAT']:
        index_params.add_index(
            field_name="vec",
            index_type=index_type,
            metric_type=cfg['metric_type'],
            params={"nlist": cfg.get('index_nlist', 128)},
            index_name=index_name
        )
    elif index_type in ['FLAT', 'BIN_FLAT']:
        index_params.add_index(
            field_name="vec",
            index_type=index_type,
            metric_type=cfg['metric_type'],
            index_name=index_name
        )
    client.create_index(collection_name, index_params)
    print(f"🔨 创建索引 {index_type}")

def load_collection(client, collection_name):
    client.load_collection(collection_name)
    print(f"📦 加载集合 {collection_name}")

def delete_collection(client, collection_name):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"🗑 删除集合 {collection_name}")

# ==================== 数据插入 ====================
def insert_data(client, collection_name, vectors, attrs=None, ids=None, batch_size=5000):
    """批量插入数据"""
    print("开始构造数据...")
    total = len(vectors)
    if ids is None:
        ids = list(range(total))
    if attrs is None:
        attrs = [1] * total
    
    print(f"构造 {total} 条数据...")
    data = [{"id": ids[i], "vec": vectors[i], "attr": attrs[i]} for i in range(total)]
    print("开始插入...")
    for i in range(0, total, batch_size):
        batch = data[i:i+batch_size]
        client.insert(collection_name, batch)
        print(f"📥 插入进度: {min(i+batch_size, total)}/{total}")
    print(f"✅ 插入完成，共 {total} 条")

# ==================== 查询测试 ====================
def search_range(client, collection_name, query_vectors, cfg, filter_expr=None, top_k=10, search_ef=64):
    """单次搜索（支持批量或单条）"""
    index_type = cfg.get('index_type', 'HNSW')
    if index_type == 'HNSW':
        search_params = {
            "metric_type": cfg['metric_type'],
            "params": {"ef": search_ef}
        }
    else:
        search_params = {
            "metric_type": cfg['metric_type'],
            "params": {"nprobe": search_ef}
        }
    start = time.time()
    res = client.search(
        collection_name=collection_name,
        data=query_vectors,
        anns_field="vec",
        search_params=search_params,
        limit=top_k,
        filter=filter_expr,
        output_fields=["id"]
    )
    elapsed = time.time() - start
    return res, elapsed

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        choices=['sift1m', 'bigann1m', 'bigann10m', 'bigann100m'],
                        help='数据集名称')
    parser.add_argument('--flat', action='store_true',
                        help='使用 FLAT 索引（暴力搜索）')
    parser.add_argument('--m', type=int, default=None, help='HNSW M 参数')
    parser.add_argument('--ef_construction', type=int, default=None, help='HNSW efConstruction 参数')
    parser.add_argument('--gamma', type=int, default=None, help='HNSW gamma 参数')
    parser.add_argument('--m_beta', type=int, default=None, help='HNSM m_beta 参数')
    parser.add_argument('--ef', type=int, default=None, help='HNSW / IVF 搜索 ef 参数')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--filter', type=str, default=None,
                        help='过滤表达式，如 "attr == 1"')
    parser.add_argument('--rebuild', action='store_true', help='强制重建集合（删除重建）')
    parser.add_argument('--rebuild_index', action='store_true', help='只重建索引（不删除数据）')
    parser.add_argument('--gt_file', type=str, default=None, help='Ground truth 文件路径')
    args = parser.parse_args()

    # 配置路径
    data_dir = f"./Datasets/{args.name}"   # 假设脚本在 Datasets 目录下运行
    cfg = DATASETS[args.name].copy()
    
    # 覆盖 HNSW 参数
    if args.m is not None:
        cfg['index_m'] = args.m
    if args.ef_construction is not None:
        cfg['index_ef_construction'] = args.ef_construction
    if args.gamma is not None:
        cfg['index_gamma'] = args.gamma
    if args.m_beta is not None:
        cfg['index_m_beta'] = args.m_beta
    if args.ef is not None:
        cfg['search_ef'] = args.ef
    
    if args.flat:
        cfg['index_type'] = 'FLAT' if cfg['vector_type'] == 'float' else 'BIN_FLAT'
        cfg['index_nlist'] = 0
        collection_name = f"{args.name}_gt"
    else:
        collection_name = args.name
    index_name = f"{collection_name}_idx"

    need_rebuild = False
    need_create_index = False
    
    # 强制重建
    client = create_client()
    if args.rebuild and client.has_collection(collection_name):
        print(f"强制重建集合 {collection_name}...")
        client.drop_collection(collection_name)
    elif args.rebuild_index and client.has_collection(collection_name):
        print(f"重建索引 {collection_name}...")
        client.drop_index(collection_name, index_name)
        need_create_index = True
    
    # 先检查集合是否存在，如果存在且数据量匹配，就跳过读取数据
    client = create_client()
    need_rebuild = False
    need_create_index = False
    
    attr_file = os.path.join(data_dir, cfg.get('attr_file', ''))
    attrs = None
    if os.path.exists(attr_file):
        print(f"读取属性: {attr_file}")
        attrs = pd.read_csv(attr_file, header=None).iloc[:, 0].tolist()
        attrs = [int(x) for x in attrs]
        print(f"属性数量: {len(attrs)}")

    if args.gt_file:
        gt_file = args.gt_file
    else:
        gt_file = os.path.join(data_dir, cfg.get('gt_file', f'{args.name}_groundtruth.ivecs'))
    if os.path.exists(gt_file):
        print(f"读取 Ground Truth: {gt_file}")
        if gt_file.endswith('.csv'):
            groundtruth = csv_gt_read(gt_file, k=args.topk)
        else:
            groundtruth = ivecs_read(gt_file, k=args.topk)
        print(f"Ground Truth 向量数: {len(groundtruth)}")
    else:
        groundtruth = None
        print("警告: 未找到 Ground Truth 文件")
    
    # 读取数据（仅在需要重建时）
    base_path = os.path.join(data_dir, cfg['base_file'])
    query_path = os.path.join(data_dir, cfg['query_file'])
    
    # 强制重建
    if args.rebuild and client.has_collection(collection_name):
        print(f"强制重建集合 {collection_name}...")
        client.drop_collection(collection_name)
    elif args.rebuild_index and client.has_collection(collection_name):
        print(f"重建索引 {collection_name}...")
        client.release_collection(collection_name)
        client.drop_index(collection_name, index_name)
        need_create_index = True
    
    if client.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在")
        stats = client.get_collection_stats(collection_name)
        row_count = stats.get('row_count', 0)
        
        # 获取期望的数据量
        with open(base_path, 'rb') as f:
            expected_count = 0
            if cfg['vector_type'] == 'float':
                f.seek(0, 2)
                file_size = f.tell()
                expected_count = file_size // (cfg['dim'] * 4 + 4)
        
        print(f"  - 数据量: {row_count}, 期望: {expected_count}")
        
        if row_count != expected_count:
            print(f"  - 数据量不匹配，需要重建")
            need_rebuild = True
            print(f"读取 base 数据: {base_path}")
            if cfg['vector_type'] == 'float':
                base_vectors = fvecs_read(base_path)
                query_vectors = fvecs_read(query_path)
            else:
                base_vectors = u8bin_read(base_path, dim_bytes=cfg['dim'])
                query_vectors = u8bin_read(query_path, dim_bytes=cfg['dim'], limit=1000)
            print(f"Base 向量数量: {len(base_vectors)}")
            print(f"Query 向量数量: {len(query_vectors)}")
        else:
            print(f"  - 数据量匹配，跳过导入")
            # 只读取 query 用于测试
            print(f"读取 query 数据: {query_path}")
            if cfg['vector_type'] == 'float':
                query_vectors = fvecs_read(query_path)
            else:
                query_vectors = u8bin_read(query_path, dim_bytes=cfg['dim'], limit=1000)
            print(f"Query 向量数量: {len(query_vectors)}")
            indexes = client.list_indexes(collection_name)
            if not indexes or len(indexes) == 0:
                print("  - 索引不存在，需要重建索引")
                need_create_index = True
            else:
                print(f"  - 索引已存在: {indexes}")
                need_create_index = False
    else:
        print(f"集合 {collection_name} 不存在，需要创建")
        need_rebuild = True
        print(f"读取 base 数据: {base_path}")
        if cfg['vector_type'] == 'float':
            base_vectors = fvecs_read(base_path)
            query_vectors = fvecs_read(query_path)
        else:
            base_vectors = u8bin_read(base_path, dim_bytes=cfg['dim'])
            query_vectors = u8bin_read(query_path, dim_bytes=cfg['dim'], limit=1000)
        print(f"Base 向量数量: {len(base_vectors)}")
        print(f"Query 向量数量: {len(query_vectors)}")

    if need_rebuild:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        create_collection(client, collection_name, cfg)
        insert_data(client, collection_name, base_vectors, attrs)
        create_index(client, collection_name, cfg, index_name)
        load_collection(client, collection_name)
    elif need_create_index:
        print("重建索引...")
        create_index(client, collection_name, cfg, index_name)
        load_collection(client, collection_name)
    else:
        if client.get_load_state(collection_name) != "Loaded":
            print("集合未加载，开始加载...")
            load_collection(client, collection_name)
        else:
            print("集合已就绪，跳过创建")

    # 执行查询性能测试
    filter_expr = args.filter

    if not filter_expr:
        filter_expr = "attr == 3"
        # if args.name == 'sift1m':
        #     filter_expr = "attr == 3"
        # elif args.name.startswith('bigann'):
        #     filter_expr = None   # 无标量过滤
        # else:
        #     filter_expr = None

    # 单条查询测试
    def run_test():
        # print("\n=== 批量搜索性能测试 ===")
        search_results, batch_time = search_range(client, collection_name, query_vectors, cfg,
                                    filter_expr, args.topk, args.ef)
        # print(f"批量搜索 {len(query_vectors)} 条总耗时: {batch_time:.3f} 秒")
        # print(f"平均每条耗时: {batch_time / len(query_vectors) * 1000:.2f} 毫秒")

        if groundtruth is not None:
            recall = compute_recall(search_results, groundtruth, args.topk)
            # print(f"召回率 @top{args.topk}: {recall*100:.2f}%")
        qps = len(query_vectors) / batch_time
        print("=====milvus=====")
        print(f"Total queries: {len(query_vectors)}")
        print(f"Total time: {batch_time:.3f} s") 
        print(f"QPS: {qps:.0f} queries/s")
        if groundtruth is not None:
            print(f"Average Recall@{args.topk}: {recall:.2f}")
        else:
            print(f"Average Recall@{args.topk}: N/A (no groundtruth)")
        return batch_time, qps, recall

    # # 逐条查询测试 (模拟真实场景)
    # print("\n=== 逐条搜索性能测试 ===")
    # single_times = []
    # for i, q in enumerate(query_vectors[:10000]):  # 只测前100条，避免太慢
    #     _, t = search_range(client, collection_name, [q], cfg,
    #                         filter_expr, args.topk, args.ef)
    #     single_times.append(t)
    #     if (i+1) % 2000 == 0:
    #         print(f"已测 {i+1} 条，平均 {np.mean(single_times)*1000:.2f} ms")
    # avg_ms = np.mean(single_times) * 1000
    # # p99_ms = np.percentile(single_times, 99) * 1000
    # num_queries = len(single_times)                     # 实际查询条数
    # total_sec = sum(single_times)                     # 总耗时（秒）
    # qps = num_queries / total_sec if total_sec > 0 else 0


    # print(f"\n总共测试 {len(single_times)} 条查询")
    # print(f"平均耗时: {avg_ms:.2f} ms")
    # print(f"P99 耗时: {p99_ms:.2f} ms")

    # # 打印总结
    # print("\n" + "="*50)
    # print("📊 测试总结")
    # print("="*50)
    # print(f"数据集: {args.name}")
    # if groundtruth is not None:
    #     print(f"召回率 @top{args.topk}: {recall*100:.2f}%")
    # print(f"单条平均耗时: {avg_ms:.2f} ms")
    # print(f"单条 QPS: {1000/avg_ms:.0f}")
    # print(f"批量平均耗时: {batch_time / len(query_vectors) * 1000:.2f} ms")
    # print(f"批量 QPS: {len(query_vectors) / batch_time:.0f}")
    # print("="*50)

    sum_time = 0
    sum_qps = 0
    sum_recall = 0
    test_times = 3
    for i in range(test_times):
        print("="*100)
        print(f"run {i+1} times")
        time, qps, recall = run_test()
        sum_time += time
        sum_qps += qps
        sum_recall += recall
    print("="*100)
    print(f"===Summary {test_times} times===")
    print(f"Ave_latency: {sum_time/test_times:.2f} s")
    print(f"Ave_QPS: {sum_qps/test_times:.0f} queries/s") 
    print(f"Ave_recall@{args.topk}: {sum_recall/test_times:.2f} ")   






if __name__ == '__main__':
    main()
    


'''
# 测试 SIFT1M
python test_milvus.py --name sift1m

# 测试 BIGANN 1M (使用 HNSW 风格索引)
python test_milvus.py --name bigann1m

# 测试 BIGANN 10M（注意内存，可能需要更多时间）
python test_milvus.py --name bigann10m --gt /data/zhupeidong/milvus/Datasets/bigann10m/gt_attr3_knn10.csv

# 使用暴力索引 (FLAT)
python test_milvus.py --name bigann1m --gt 1
'''