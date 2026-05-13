import argparse
import os
import struct
import sys
import tempfile
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp


EDGE_EXT = ".bin"
SPARSE_EXT = ".npz"


def parse_args():
    parser = argparse.ArgumentParser(description="分块稀疏 PageRank")
    parser.add_argument("--data", default="Data.txt", help="输入边文件")
    parser.add_argument("--out", default="Res.txt", help="输出文件")
    parser.add_argument("--teleport", type=float, default=0.85, help="阻尼/跳转参数")
    parser.add_argument("--tol", type=float, default=1e-8, help="收敛阈值")
    parser.add_argument("--max-iter", type=int, default=100, help="最大迭代次数")
    parser.add_argument("--block-size", type=int, default=5000, help="每个块包含的节点数量")
    parser.add_argument("--max-open", type=int, default=64, help="同时打开的块文件数")
    parser.add_argument("--rebuild", action="store_true", help="重新构建块文件和稀疏矩阵")
    return parser.parse_args()


def read_data(file_path):
    """读取边数据，返回节点映射、边列表和出度统计"""
    edges = []
    nodes = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = map(int, parts[:2])
            except ValueError:
                continue
            if u < 0 or v < 0:
                continue
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    if not edges:
        raise ValueError("输入文件里没有有效边数据")

    node_list = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)

    out_degree = np.zeros(n, dtype=np.int32)
    for u, _ in edges:
        out_degree[node_to_idx[u]] += 1

    return edges, node_list, node_to_idx, out_degree


class BlockWriter:
    def __init__(self, block_dir, max_open):
        self.block_dir = block_dir
        self.max_open = max_open
        self.handles = OrderedDict()

    def _open(self, block_id):
        if block_id in self.handles:
            self.handles.move_to_end(block_id)
            return self.handles[block_id]

        if len(self.handles) >= self.max_open:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()

        path = os.path.join(self.block_dir, f"block_{block_id}{EDGE_EXT}")
        handle = open(path, "ab")
        self.handles[block_id] = handle
        return handle

    def write_edge(self, block_id, src_idx, dst_idx):
        handle = self._open(block_id)
        handle.write(struct.pack("<ii", src_idx, dst_idx))

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def build_block_files(edges, node_to_idx, block_dir, block_size, max_open):
    """把边按源节点分块写到磁盘，减少内存占用"""
    os.makedirs(block_dir, exist_ok=True)
    writer = BlockWriter(block_dir, max_open)

    for u, v in edges:
        src_idx = node_to_idx[u]
        dst_idx = node_to_idx[v]
        block_id = src_idx // block_size
        writer.write_edge(block_id, src_idx, dst_idx)

    writer.close()


def build_sparse_blocks(block_dir, sparse_dir, block_size, n, out_degree):
    """把每个块转成 scipy 的稀疏矩阵，后面迭代时直接做矩阵乘法"""
    os.makedirs(sparse_dir, exist_ok=True)

    block_files = sorted(
        os.path.join(block_dir, name)
        for name in os.listdir(block_dir)
        if name.endswith(EDGE_EXT)
    )

    for block_path in block_files:
        name = os.path.basename(block_path)
        block_id = int(name[len("block_") : -len(EDGE_EXT)])

        data = np.fromfile(block_path, dtype=np.int32)
        if data.size == 0:
            continue

        data = data.reshape(-1, 2)
        src = data[:, 0]
        dst = data[:, 1]

        block_start = block_id * block_size
        block_end = min(block_start + block_size, n)
        block_width = block_end - block_start

        # 这里是列归一化的块矩阵，每一列对应一个源节点
        cols = src - block_start
        weights = (1.0 / out_degree[src]).astype(np.float32)
        block_matrix = sp.csr_matrix(
            (weights, (dst, cols)), shape=(n, block_width), dtype=np.float32
        )

        out_path = os.path.join(sparse_dir, f"block_{block_id}{SPARSE_EXT}")
        sp.save_npz(out_path, block_matrix)


def pagerank(sparse_dir, n, block_size, out_degree, teleport=0.85, tol=1e-8, max_iter=100):
    """PageRank 主循环，处理 dead-end 和 spider-trap，直到收敛"""
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    dangling_mask = out_degree == 0

    block_files = sorted(
        os.path.join(sparse_dir, name)
        for name in os.listdir(sparse_dir)
        if name.endswith(SPARSE_EXT)
    )

    for it in range(max_iter):
        old_pr = pr.copy()

        # dead-end 节点会把它们的权重平均传回全图
        dangling_weight = old_pr[dangling_mask].sum(dtype=np.float64)
        base = (1.0 - teleport) / n + teleport * dangling_weight / n
        new_pr = np.full(n, base, dtype=np.float64)

        for block_path in block_files:
            name = os.path.basename(block_path)
            block_id = int(name[len("block_") : -len(SPARSE_EXT)])
            block_matrix = sp.load_npz(block_path)

            block_start = block_id * block_size
            block_end = block_start + block_matrix.shape[1]
            block_pr = old_pr[block_start:block_end]

            # 块矩阵乘法，保留了 block matrix + sparse matrix 的思路
            new_pr += teleport * (block_matrix @ block_pr)

        diff = np.linalg.norm(new_pr - old_pr, 1)
        pr = new_pr

        if diff < tol:
            print(f"Converged in {it + 1} iterations")
            break

    return pr


def save_top_100(pr, node_list, output_file="Res.txt"):
    idx_sorted = np.argsort(-pr)[: min(100, len(pr))]
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in idx_sorted:
            f.write(f"{node_list[idx]} {pr[idx]:.15f}\n")


def main():
    args = parse_args()
    base_dir = os.path.dirname(__file__)

    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)

    if not os.path.exists(data_path):
        print(f"找不到输入文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    # 第一步：读入数据，建立节点映射，只保留真正出现过的 NodeID
    edges, node_list, node_to_idx, out_degree = read_data(data_path)
    n = len(node_list)

    # 第二步：构建分块文件和稀疏矩阵，方便后续迭代时节省内存
    # 这些中间文件放到临时目录里，程序结束后自动清理，不留多余产物
    with tempfile.TemporaryDirectory(prefix="pagerank_blocks_", dir=base_dir) as work_dir:
        block_dir = os.path.join(work_dir, "edge_blocks")
        sparse_dir = os.path.join(work_dir, "sparse_blocks")

        build_block_files(edges, node_to_idx, block_dir, args.block_size, args.max_open)
        build_sparse_blocks(block_dir, sparse_dir, args.block_size, n, out_degree)

        # 第三步：PageRank 迭代
        pr = pagerank(
            sparse_dir,
            n,
            args.block_size,
            out_degree,
            teleport=args.teleport,
            tol=args.tol,
            max_iter=args.max_iter,
        )

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(base_dir, out_path)
    save_top_100(pr, node_list, out_path)


if __name__ == "__main__":
    main()
