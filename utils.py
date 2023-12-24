import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def loadData(pathPrefix: str, name: str):
    pcd_1 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-1.ply")
    pcd_1_points = np.asarray(pcd_1.points)
    pcd_2 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-2.ply")
    pcd_1_points = np.asarray(pcd_1.points)
    pcd_3 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-3.ply")
    pcd_1_points = np.asarray(pcd_1.points)

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

    return matches

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