# 暂时用不上的函数
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 找到矩阵中最小的几个对应点
def findMinInMatrix(matrix, num):
    idx1 = []
    idx2 = []
    visited_row = set()
    visited_col = set()
    for _ in range(num):
        # find minimum
        mini = [0, 0, float("inf")]
        for i in range(matrix.shape[0]):
            if i in visited_row:
                continue
            for j in range(matrix.shape[1]):
                if j in visited_col:
                    continue
                cur = matrix[i][j]
                if cur < mini[2]:
                    mini = [i, j, cur]
        idx1.append(mini[0])
        idx2.append(mini[1])
        visited_row.add(mini[0])
        visited_col.add(mini[1])
        return np.array(idx1), np.array(idx2)
    

# 找最近邻 (这里src和tgt反了)
def find_n(src, tgt):
    nk = len(src)
    neighbors = NearestNeighbors(n_neighbors=nk, algorithm="kd_tree").fit(tgt)
    dists, idxs = neighbors.kneighbors(src)
    # 可用数量
    # val_num = int(np.floor(nk * 0.25))
    val_num = int(np.floor(nk * 0.20))
    srcid = np.argsort(dists[:, 0])
    tgtid = idxs[srcid, 0]
    tgtid_, indices = pro(tgtid, val_num)
    srcid_ = srcid[indices]
    return [srcid_, tgtid_]


def pro(lst, n):
    seen = set()
    result = []
    indices = []
    for i, num in enumerate(lst):
        if len(result) == n:
            break
        if num not in seen:
            seen.add(num)
            result.append(num)
            indices.append(i)
    return result, indices