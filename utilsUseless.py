# 暂时用不上的函数
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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

def find_nn_kdTree(src: np.ndarray, tgt: np.ndarray):
    """
    Find one-to-one correspondence between points in source and target point clouds to minimize the sum of Euclidean distances.

    Parameters:
        src (np.ndarray): Source point cloud, Nx3 array.
        tgt (np.ndarray): Target point cloud, Mx3 array.

    Returns:
        src_indices (np.ndarray): Indices of corresponding points in the source cloud.
        tgt_indices (np.ndarray): Indices of corresponding points in the target cloud.
    """
    # Build KD tree for the target point cloud
    tree = cKDTree(tgt)

    # Query the KD tree to find the nearest neighbors for each point in the source cloud
    _, indices = tree.query(src, k=1)

    # Return the indices of corresponding points
    src_indices = np.arange(src.shape[0])
    tgt_indices = np.asarray(indices.flatten())

    return np.asarray([src_indices, tgt_indices])

def find_nn_posAndFeat(points1, points2, feat1, feat2, pos_w, feat_w, k):
    # 计算两个数组中所有点的距离矩阵
    pos_dis = cdist(points1, points2)
    feat_dis = cdist(feat1, feat2)
    distances = pos_w * pos_dis + feat_w * feat_dis

    # 初始化匹配对列表
    matches = []
    # 找到最近邻匹配对
    for _ in range(k):
        # 找到最小距离的索引
        idx = np.unravel_index(np.argmin(distances), distances.shape)
        # 添加匹配对到列表
        matches.append((idx[0], idx[1]))
        # 将已匹配的点的距离设为无穷大，以避免重复匹配
        distances[idx[0], :] = np.inf
        distances[:, idx[1]] = np.inf

    return np.asarray(matches)