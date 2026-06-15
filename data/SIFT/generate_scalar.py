import numpy as np
import h5py

# 1. 读取 SIFT HDF5 文件，获取数据总量
with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    n_total = f['train'].shape[0]  # 假设向量存储在 'train' 键下
    print(f"文件中共有 {n_total} 条向量")

# 2. 生成一个与数据量大小相同的标量数组（例如，全 0 的 uint16 数组）
# 注意：dtype 必须与函数期望的 'H' (np.uint16) 类型匹配
scalar_data = np.zeros((n_total,), dtype=np.uint16)
# 如果需要更真实的测试数据，可以用随机数，但全零是合法的
# scalar_data = np.random.randint(0, 65535, size=(n_total,), dtype=np.uint16)

# 3. 将数组保存为 VecBench 需要的二进制格式
def save_scalar_bin(arr, filename):
    with open(filename, 'wb') as f:
        f.write(np.int32(arr.shape[0]).tobytes())  # 写入数量 (n)
        f.write(np.int32(1).tobytes())           # 写入维度 (dim)，必须是 1
        arr.tofile(f)                             # 写入数组数据

save_scalar_bin(scalar_data, 'sift_scalar.bin')
print("文件 'sift_scalar.bin' 已生成")