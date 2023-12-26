import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
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


def find_nn(points1, points2, k=None):
    if not k:
        k = min(points1.shape[0], points2.shape[0])
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


def com_loss(A, B):
    d = np.linalg.norm(A - B, axis=1)  # 计算每一行的距离
    sum_d = np.sum(d)
    return sum_d / len(A)


def sortAndShow(corr, s=True):
    corrIdx_ = copy.deepcopy(corr)
    if s:
        sorted_indices = np.argsort(corrIdx_[:, 0])
        corrIdx_ = corrIdx_[sorted_indices]
    print(f"src_idx: {corrIdx_[:, 0]}")
    print(f"tgt_idx: {corrIdx_[:, 1]}")


def corrSubPcd(pcd1, pcd2):
    """
    找对应的子点云
    先验：
        pcd1, pcd2 是有序的，根据 distance(pcd1_feature, pcd2_feature) 从小到大排序
        所以 pcd1[0] 和 pcd2[0] 极有可能是对应点
    """
    n = pcd1.shape[0]  # 也等于 pcd2.shape[0]
    dist1, dist2 = cdist(pcd1, pcd1), cdist(pcd2, pcd2)

    threshold = 6e-3

    for i in range(n):
        ne = n - i - 1
        # for p in range(ne):
        #     for q in range(ne):
        #         dis_err[p, q] = abs(dist1[i, p + i + 1] - dist2[i, q + i + 1])
        dis_err = np.abs(dist1[i, i + 1 :] - dist2[i, i + 1 :])
        idx = np.where(dis_err < threshold)[0]

        if idx.size > 0:
            idx = idx + i + 1
            idx = idx.astype(int)
            return np.insert(idx, 0, i)
        # coordinates = np.vstack(np.where(dis_err < threshold)).T
        # 如果大于0，这些对应点就是要找的点
        # if coordinates.size() > 0:
        #     # 有映射关系
        #     coordinates = coordinates + i + 1
        #     return coordinates

    return np.asarray([])
