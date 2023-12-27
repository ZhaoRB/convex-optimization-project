import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist


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


def findCorrSubPcd(pcd1, pcd2, threshold):
    """
    找对应的子点云
    先验：
        pcd1, pcd2 是有序的，根据 distance(pcd1_feature, pcd2_feature) 从小到大排序
        所以 pcd1[0] 和 pcd2[0] 极有可能是对应点
    """
    n = pcd1.shape[0]  # 也等于 pcd2.shape[0]
    dist1, dist2 = cdist(pcd1, pcd1), cdist(pcd2, pcd2)

    for i in range(n):
        dis_err = np.abs(dist1[i, i + 1 :] - dist2[i, i + 1 :])
        idx = np.where(dis_err < threshold)[0]

        if idx.size > 0:
            idx = idx + i + 1
            idx = idx.astype(int)
            return np.insert(idx, 0, i)

    return np.asarray([])


def saveAsPly(points: np.ndarray, path):
    pcd = npToPcd(points)
    o3d.io.write_point_cloud(path, pcd)


def saveVisibleResults(pcds: list[np.ndarray], path=None, isVisible=False):
    n = len(pcds)
    labels = ["Registrated", "Target", "Source"]
    colors = ["red", "blue", "green"]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the registered and target point clouds
    for i in range(n):
        ax.scatter(
            pcds[i][:, 0],
            pcds[i][:, 1],
            pcds[i][:, 2],
            c=colors[i],
            label=labels[i],
            s=0.1,
        )

    # Set labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if path is not None:
        plt.savefig(path)
    if isVisible:
        plt.show()
