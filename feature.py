import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist


# 使用FPFH算法计算点云中指定的salient points的特征
def featureExtraction(
    pointCloud, salientPointsIndex, hyperparams, isVisualize=False
) -> np.ndarray:
    # 计算法向量
    pointCloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=hyperparams["r_normal"], max_nn=hyperparams["max_nn_norm"]
        )
    )

    # 构建 FPFH 特征估计器
    fpfh_estimator = o3d.pipelines.registration.compute_fpfh_feature(
        pointCloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=hyperparams["r_fpfh"], max_nn=hyperparams["max_nn_fpfh"]
        ),
    )

    # 获取 FPFH 特征
    key_points_fpfh = fpfh_estimator.data[:, salientPointsIndex]

    if isVisualize:
        print(f"features shape: {key_points_fpfh.shape}")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(key_points_fpfh.shape[1]):
            ax.plot(key_points_fpfh[:, i], label=f"Key Point {i + 1}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("FPFH Value")
        ax.legend()
        plt.show()

    return key_points_fpfh.T


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
