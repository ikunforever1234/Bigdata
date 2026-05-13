import os
import sys

import numpy as np

# 可直接手动改这里
DATA_FILE = "Data.txt"
OUTPUT_FILE = "Res.txt"
TELEPORT = 0.85
TOL = 1e-8
MAX_ITER = 100
DTYPE = np.float64


def read_data_dense(file_path):
    """读取边数据，构建节点映射与边索引。"""
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
        raise ValueError("输入文件中没有有效边")

    node_list = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)

    src_idx = np.fromiter((node_to_idx[u] for u, _ in edges), dtype=np.int32, count=len(edges))
    dst_idx = np.fromiter((node_to_idx[v] for _, v in edges), dtype=np.int32, count=len(edges))

    out_degree = np.zeros(n, dtype=np.int32)
    np.add.at(out_degree, src_idx, 1)

    return node_list, src_idx, dst_idx, out_degree, n


def build_dense_transition(n, src_idx, dst_idx, out_degree):
    """构建列随机密集转移矩阵 S，dead-end 列填 1/n。"""
    s = np.zeros((n, n), dtype=DTYPE)

    non_dangling = out_degree[src_idx] > 0
    src = src_idx[non_dangling]
    dst = dst_idx[non_dangling]

    weights = (1.0 / out_degree[src]).astype(DTYPE)
    np.add.at(s, (dst, src), weights)

    dangling_mask = out_degree == 0
    if np.any(dangling_mask):
        s[:, dangling_mask] = 1.0 / n

    return s


def pagerank_dense_power(s, teleport=0.85, tol=1e-8, max_iter=100):
    """密集矩阵幂迭代求解 PageRank。"""
    n = s.shape[0]
    pr = np.full(n, 1.0 / n, dtype=DTYPE)

    for _ in range(max_iter):
        old_pr = pr
        pr = (1.0 - teleport) / n + teleport * (s @ old_pr)

        diff = float(np.linalg.norm(pr - old_pr, 1))
        if diff < tol:
            break

    total = float(pr.sum(dtype=np.float64))
    if total > 0:
        pr /= total
    return pr


def save_top_100(pr, node_list, output_file):
    """保存 Top-100：NodeID Score"""
    top_k = min(100, len(pr))
    idx = np.argsort(-pr)[:top_k]
    with open(output_file, "w", encoding="utf-8") as f:
        for i in idx:
            f.write(f"{node_list[i]} {pr[i]:.15f}\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = DATA_FILE
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)
    if not os.path.exists(data_path):
        print(f"找不到输入文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    node_list, src_idx, dst_idx, out_degree, n = read_data_dense(data_path)
    s = build_dense_transition(n, src_idx, dst_idx, out_degree)
    pr = pagerank_dense_power(s, teleport=TELEPORT, tol=TOL, max_iter=MAX_ITER)

    out_path = OUTPUT_FILE
    if not os.path.isabs(out_path):
        out_path = os.path.join(base_dir, out_path)
    save_top_100(pr, node_list, out_path)


if __name__ == "__main__":
    main()
