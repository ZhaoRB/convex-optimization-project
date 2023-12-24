import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from solver import *
from utils import *


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

    return key_points_fpfh


# 使用feature过滤salient points
def featureFilter(features1, features2, ratio=2):
    row = features1.shape[1]
    col = features2.shape[1]

    num = min(row, col) // ratio
    idx_features1 = np.zeros(num, dtype=int)
    idx_features2 = np.zeros(num, dtype=int)
    dis = []

    distances_matrix = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            distances_matrix[i][j] = np.linalg.norm(features1[:, i] - features2[:, j])

    visited_row = set()
    visited_col = set()
    for idx in range(num):
        # find minimum
        mini = [0, 0, float("inf")]
        for i in range(row):
            if i in visited_row:
                continue
            for j in range(col):
                if j in visited_col:
                    continue
                cur = distances_matrix[i][j]
                if cur < mini[2]:
                    mini = [i, j, cur]
        idx_features1[idx] = mini[0]
        idx_features2[idx] = mini[1]
        visited_row.add(mini[0])
        visited_col.add(mini[1])
        dis.append(mini[2])

    print("least distance points:")
    print(idx_features1)
    print(idx_features2)
    print(dis)

    return np.array([idx_features1, idx_features2])


# global registration
def globalReg(src_pcd, tgt_pcd, src_feat, tgt_feat):
    # 根据特征值找对应点
    # corr = featureFilter(src_feat, tgt_feat)
    # corr = np.asarray(find_n(src_feat.T, tgt_feat.T))
    corr = np.asarray(find_n(tgt_feat.T, src_feat.T))
    src = np.asarray(src_pcd.points)[corr[1]]
    tgt = np.asarray(tgt_pcd.points)[corr[0]]

    # 粗配准
    cen_source = np.mean(src, axis=0)
    cen_target = np.mean(tgt, axis=0)
    # 计算零中心配对点对
    cen_cor_source = src - cen_source
    cen_cor_tar = tgt - cen_target
    # 使用奇异值分解（SVD）估计旋转矩阵
    H = np.dot(cen_cor_source.T, cen_cor_tar)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # 计算平移向量
    T = cen_source - np.dot(R, cen_target)
    return R, T


def pointCloudRegistration(prefix, name, hyperparams):
    # 1. load data
    pcds, infos = loadData(prefix, name)

    # 2. feature extraction
    salient_features = [
        featureExtraction(pcd, info["all_idxs"], hyperparams["fpfh"])
        for pcd, info in zip(pcds, infos)
    ]

    tgt_pcd = pcds[0]
    tgt_pcd_salient = tgt_pcd.select_by_index(infos[0]["all_idxs"])
    tgt_salient_feature = salient_features[0]

    for i in range(2):
        idx = i + 1
        src_pcd = pcds[idx]
        src_pcd_salient = src_pcd.select_by_index(infos[idx]["all_idxs"])
        src_salient_feature = salient_features[idx]

        # 3. global registration
        R, T = globalReg(
            src_pcd_salient, tgt_pcd_salient, src_salient_feature, tgt_salient_feature
        )

        print(R)
        print(T)

        # transform
        golReg_pcd = o3d.geometry.PointCloud()
        reg_points = np.transpose(R @ np.asarray(src_pcd.points).T + T.reshape(3, 1))
        golReg_pcd.points = o3d.utility.Vector3dVector(reg_points)
        golReg_pcd_salient = golReg_pcd.select_by_index(infos[idx]["all_idxs"])

        # visualization
        src_pcd.paint_uniform_color([1, 0, 0])  # 红
        tgt_pcd.paint_uniform_color([0, 1, 0])  # 绿
        golReg_pcd.paint_uniform_color([0, 0, 1])  # 蓝
        o3d.visualization.draw_geometries([src_pcd, tgt_pcd, golReg_pcd])

        # 4. icp local registration
        # R, T = icp(
        #     np.asarray(src_pcd_salient.points),
        #     np.asarray(tgt_pcd_salient.points),
        #     np.asarray(src_pcd.points),
        #     np.asarray(tgt_pcd.points),
        #     src_salient_feature,
        #     tgt_salient_feature,
        #     R,
        #     T,
        #     0.1,
        # )


if __name__ == "__main__":
    # names = ["bunny", "room", "temple"]
    names = ["bunny"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    hyperparams = [
        {
            "fpfh": {
                "r_normal": 0.02,
                "r_fpfh": 0.02,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "ransac": {"dis": 0.1},
        }
    ]
    for idx, name in enumerate(names):
        pointCloudRegistration(f"{prefix}/{name}-pcd", name, hyperparams[idx])
