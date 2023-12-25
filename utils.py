import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def loadData(pathPrefix: str, name: str):
    pcd_1 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-1.ply")
    pcd_2 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-2.ply")
    pcd_3 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-3.ply")

    json_path = f"{pathPrefix}/points.json"
    with open(json_path, "r") as file:
        info = json.load(file)
    info_1 = info["pcd_1"]
    info_2 = info["pcd_2"]
    info_3 = info["pcd_3"]

    return [pcd_1, pcd_2, pcd_3], [info_1, info_2, info_3]


def find_nn(points1, points2, k):
    # 计算两个数组中所有点的距离矩阵
    distances = cdist(points1, points2)

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


def compare_pcd(pcds, labels=None, path=None):
    if labels is None:
        labels = ["Source", "Registrated", "Target"]

    dpi = 80
    fig = plt.figure(figsize=(1440 / dpi, 720 / dpi), dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("persp")

    for points, label in zip(pcds, labels):
        ax.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            marker=".",
            alpha=0.5,
            edgecolors="none",
            label=label,
        )

    plt.legend()
    plt.show()
    #
    # plt.clf()
    if path is not None:
        plt.savefig(f"./imgs/{path}.png")


# 计算欧式距离（特征 or 位置）+ 归一化
def com_sim(srcMat: np.ndarray, tgtMat: np.ndarray):  # scrMat 33x33    tgtMat 36x33
    m = srcMat.shape[0]
    n = tgtMat.shape[0]
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            res[i, j] = np.linalg.norm(srcMat[i, :] - tgtMat[j, :])
    # 归一化
    return res / np.amax(res)


def pcdToNp(pointCloud):
    return np.asarray(pointCloud.points)


def npToPcd(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd


def pcd_visualize(point_collections: list[np.ndarray]):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    merged_pcd = o3d.geometry.PointCloud()

    for i, points in enumerate(point_collections):
        # create PointCloud object & convert ndarray to points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # set color
        pcd.paint_uniform_color(colors[i])
        merged_pcd += pcd

    o3d.visualization.draw_geometries([merged_pcd])


def find_nn_corr(src, tgt):
    """Given two input point clouds, find nearest-neighbor correspondence (from source to target)
    Input:
        - src: Source point cloud (n*3), either array or open3d pcd
        - tgt: Target point cloud (n*3), either array or open3d pcd
    Output:
        - idxs: Array indices corresponds to src points,
            array elements corresponds to nn in tgt points (n, np.array)
    """

    """ Way1: Sklearn"""
    if src.shape[1] != 3:
        src = src.T
    if tgt.shape[1] != 3:
        tgt = tgt.T

    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)  # (16384*3)
        tgt = np.asarray(tgt.points)

    n1 = src.shape[0]
    n2 = tgt.shape[0]
    if n1 < n2:
        tgt = tgt[:n1]
    else:
        src = src[:n2]

    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(tgt)
    dists, idxs = neighbors.kneighbors(src)  # (16384*1), (16384*1)
    return np.asarray([range(min(n1, n2)), idxs])

# src, tgt 分别代表两个点云的数据，点云数量不一定相等
# 目标：寻找两个点云一对一的对应关系，使得点对的欧式距离之和最小
# 返回：src和tgt的idx
def find_min_sum(src: np.ndarray, tgt: np.ndarray):
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

    return src_indices, tgt_indices


def com_loss(A, B):
    d = np.linalg.norm(A - B, axis=1)  # 计算每一行的距离
    sum_d = np.sum(d)
    return sum_d / len(A)