import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils import *


# 使用FPFH算法计算点云中指定的salient points的特征
def featureExtraction(pointCloud, salientPointsIndex, hyperparams, isVisualize=False):
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
def featureFilter(features1, features2, points1, points2):
    row = features1.shape[1]
    col = features2.shape[1]

    num = min(row, col) // 2
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

    # filtered points
    filtered_points1 = points1.select_by_index(idx_features1)
    filtered_points2 = points2.select_by_index(idx_features2)

    return (
        features1[:, idx_features1],
        features2[:, idx_features2],
        filtered_points1,
        filtered_points2,
    )


def featureMatching(src, tgt, src_fpfh, tgt_fpfh, hyperparams):
    src_features = src_fpfh.data
    tgt_features = tgt_fpfh.data

    # 过滤掉一些不相关的点
    src_features, tgt_features, src, tgt = featureFilter(
        src_features, tgt_features, src, tgt
    )

    distance_threshold = hyperparams["dis"]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            True
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def localRegistration(src, tgt, src_fpfh, tgt_fpfh, hyperparams):
    pass


def pointCloudRegistration(prefix, name, hyperparams):
    # 1. load data
    pcds, infos = loadData(prefix, name)

    # 2. feature extraction
    features = [
        featureExtraction(pcd, info["all_idxs"], hyperparams["fpfh"])
        for pcd, info in zip(pcds, infos)
    ]

    tgt_pcd = pcds[0]
    tgt_pcd_salient = tgt_pcd.select_by_index(infos[0]["all_idxs"])
    tgt_fpfh = o3d.pipelines.registration.Feature()
    tgt_fpfh.data = features[0]

    for i in range(2):
        idx = i + 1
        src_pcd = pcds[idx]
        src_pcd_salient = src_pcd.select_by_index(infos[idx]["all_idxs"])
        src_fpfh = o3d.pipelines.registration.Feature()
        src_fpfh.data = features[idx]

        # 3. feature matching & global registration
        res = featureMatching(
            src_pcd_salient, tgt_pcd_salient, src_fpfh, tgt_fpfh, hyperparams["ransac"]
        )
        corr_set = np.asarray(res.correspondence_set)  # ransac 找到的对应关系
        trans = res.transformation  # 得到的 transform 矩阵（4*4）

        print(
            f"feature corresponsence set: \nshape: {corr_set.shape} \n values: \n{corr_set.T}"
        )

        # transform
        pcd_reg = o3d.geometry.PointCloud()
        pcd_reg.points = src_pcd.points
        pcd_reg.transform(trans)

        # visualization
        src_pcd.paint_uniform_color([1, 0, 0])  # 红
        tgt_pcd.paint_uniform_color([0, 1, 0])  # 绿
        pcd_reg.paint_uniform_color([0, 0, 1])  # 蓝
        o3d.visualization.draw_geometries([src_pcd, tgt_pcd, pcd_reg])

        # 4. icp local registration

if __name__ == "__main__":
    # names = ["bunny", "room", "temple"]
    names = ["bunny"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    hyperparams = [
        {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "ransac": {"dis": 0.1},
        }
    ]
    for idx, name in enumerate(names):
        pointCloudRegistration(f"{prefix}/{name}-pcd", name, hyperparams[idx])
