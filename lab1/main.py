import gc
import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

# 可直接手动改这里
DATA_FILE = "Data.txt"
OUTPUT_FILE = "Res.txt"
TELEPORT = 0.85
TOL = 1e-8
MAX_ITER = 100
BLOCK_SIZE = 5000


def get_app_dir():
    """返回程序运行目录：普通脚本时为源码目录，打包后为 exe 所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resolve_input_path(file_name):
    """优先找 exe 同目录下的外部文件，再回退到 PyInstaller 解包目录。"""
    app_dir = get_app_dir()
    candidate = file_name if os.path.isabs(file_name) else os.path.join(app_dir, file_name)
    if os.path.exists(candidate):
        return candidate

    if getattr(sys, "frozen", False):
        bundled = os.path.join(getattr(sys, "_MEIPASS", app_dir), file_name)
        if os.path.exists(bundled):
            return bundled

    return candidate


def resolve_output_path(file_name):
    """输出始终写到 exe 同目录，便于在其他电脑上直接查看。"""
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(get_app_dir(), file_name)


def read_and_build_blocks(file_path, block_size):
    """两遍读取：先统计，再按块写边文件。"""
    nodes = set()
    out_deg_dict = {}

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
            nodes.add(u)
            nodes.add(v)
            out_deg_dict[u] = out_deg_dict.get(u, 0) + 1

    if not nodes:
        raise ValueError("输入文件中没有有效边")

    node_list = sorted(nodes)
    n = len(node_list)

    node_to_idx = {node: i for i, node in enumerate(node_list)}
    out_degree = np.zeros(n, dtype=np.int32)
    for node, deg in out_deg_dict.items():
        out_degree[node_to_idx[node]] = deg

    del nodes, out_deg_dict
    gc.collect()

    block_dir = tempfile.mkdtemp(prefix="pr_blocks_")
    num_blocks = (n + block_size - 1) // block_size
    buffers = [[] for _ in range(num_blocks)]
    flush_limit = 50000

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

            src_idx = node_to_idx[u]
            dst_idx = node_to_idx[v]
            bid = src_idx // block_size
            buffers[bid].append((src_idx, dst_idx))

            if len(buffers[bid]) >= flush_limit:
                arr = np.array(buffers[bid], dtype=np.int32)
                with open(os.path.join(block_dir, f"block_{bid}.bin"), "ab") as fh:
                    fh.write(arr.tobytes())
                buffers[bid].clear()

    for bid, buf in enumerate(buffers):
        if buf:
            arr = np.array(buf, dtype=np.int32)
            with open(os.path.join(block_dir, f"block_{bid}.bin"), "ab") as fh:
                fh.write(arr.tobytes())

    del node_to_idx, buffers
    gc.collect()

    return node_list, out_degree, block_dir, n


def build_sparse_blocks(block_dir, sparse_dir, block_size, n, out_degree):
    """把每个 .bin 块转成列归一化 CSR 稀疏矩阵。"""
    os.makedirs(sparse_dir, exist_ok=True)

    for fname in sorted(os.listdir(block_dir)):
        if not fname.endswith(".bin"):
            continue

        fpath = os.path.join(block_dir, fname)
        block_id = int(fname[len("block_") : -len(".bin")])

        data = np.fromfile(fpath, dtype=np.int32)
        if data.size == 0:
            continue

        data = data.reshape(-1, 2)
        src = data[:, 0]
        dst = data[:, 1]

        block_start = block_id * block_size
        block_end = min(block_start + block_size, n)
        block_width = block_end - block_start

        cols = src - block_start
        weights = (1.0 / out_degree[src]).astype(np.float32)

        mat = sp.csr_matrix(
            (weights, (dst, cols)),
            shape=(n, block_width),
            dtype=np.float32,
        )

        out_path = os.path.join(sparse_dir, f"block_{block_id}.npz")
        sp.save_npz(out_path, mat)

        del data, src, dst, cols, weights, mat

    for fname in os.listdir(block_dir):
        if fname.endswith(".bin"):
            os.unlink(os.path.join(block_dir, fname))

    gc.collect()


def pagerank(sparse_dir, n, block_size, out_degree, teleport=0.85, tol=1e-8, max_iter=100):
    """PageRank 迭代（含 dead-end 处理）。"""
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    dangling_mask = out_degree == 0

    block_paths = sorted(
        os.path.join(sparse_dir, f)
        for f in os.listdir(sparse_dir)
        if f.endswith(".npz")
    )

    for _ in range(max_iter):
        old_pr = pr.copy()

        dangling_weight = float(old_pr[dangling_mask].sum(dtype=np.float64))
        base = (1.0 - teleport) / n + teleport * dangling_weight / n
        new_pr = np.full(n, base, dtype=np.float64)

        for bp in block_paths:
            block_id = int(os.path.basename(bp)[len("block_") : -len(".npz")])
            mat = sp.load_npz(bp)
            start = block_id * block_size
            end = start + mat.shape[1]
            new_pr += teleport * (mat @ old_pr[start:end])
            del mat

        diff = float(np.linalg.norm(new_pr - old_pr, 1))
        pr = new_pr
        if diff < tol:
            break

    return pr


def save_top_100(pr, node_list, output_file):
    """保存 Top-100：NodeID Score"""
    idx = np.argsort(-pr)[:100]
    with open(output_file, "w", encoding="utf-8") as f:
        for i in idx:
            f.write(f"{node_list[i]} {pr[i]:.15f}\n")


def main():
    base_dir = get_app_dir()

    data_path = resolve_input_path(DATA_FILE)
    if not os.path.exists(data_path):
        print(f"找不到输入文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    node_list, out_degree, block_dir, n = read_and_build_blocks(data_path, BLOCK_SIZE)

    sparse_dir = tempfile.mkdtemp(prefix="pr_sparse_")
    try:
        build_sparse_blocks(block_dir, sparse_dir, BLOCK_SIZE, n, out_degree)

        for f in os.listdir(block_dir):
            os.unlink(os.path.join(block_dir, f))
        os.rmdir(block_dir)

        pr = pagerank(
            sparse_dir,
            n,
            BLOCK_SIZE,
            out_degree,
            teleport=TELEPORT,
            tol=TOL,
            max_iter=MAX_ITER,
        )

    finally:
        for f in os.listdir(sparse_dir):
            os.unlink(os.path.join(sparse_dir, f))
        os.rmdir(sparse_dir)

    out_path = resolve_output_path(OUTPUT_FILE)
    save_top_100(pr, node_list, out_path)


if __name__ == "__main__":
    main()
