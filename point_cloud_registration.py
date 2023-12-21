import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils import *


# 使用FPFH算法计算点云中指定的salient points的特征
def featureExtraction(
    pointCloud, salientPointsIndex, radius_normal=0.25, radius_fpfh=0.25
):
    # 计算法向量 也需要在邻域中计算
    pointCloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # 构建 FPFH 特征估计器（只能先计算所有的特征，再提取出salient points的特征）
    fpfh_estimator = o3d.pipelines.registration.compute_fpfh_feature(
        pointCloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_fpfh, max_nn=100),
    )

    # 获取 FPFH 特征
    key_points_fpfh = fpfh_estimator.data[:, salientPointsIndex]

    print(f"features shape: {key_points_fpfh.shape}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(key_points_fpfh.shape[1]):
        ax.plot(key_points_fpfh[:, i], label=f"Key Point {i + 1}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("FPFH Value")
    ax.legend()
    plt.show()

    return key_points_fpfh


def featureMatching(src, tgt, src_fpfh, tgt_fpfh):
    distance_threshold = 0.05

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        ransac_n=3,
        # checkers = [
        #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
        #         distance_threshold
        #     ),
        # ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def pointCloudRegistration(prefix, name):
    # hyperparameters
    radius_normal = 0.25
    radius_fpfh = 0.25

    # 1. load data
    pcds, infos = loadData(prefix, name)

    # 2. feature extraction
    features = [
        featureExtraction(pcd, info["all_idxs"]) for pcd, info in zip(pcds, infos)
    ]

    # 3. feature matching & global registration
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

        # matching
        res = featureMatching(src_pcd_salient, tgt_pcd_salient, src_fpfh, tgt_fpfh)
        corr_set = np.asarray(res.correspondence_set)  # 对应关系
        trans = res.transformation  # 得到的 transform 矩阵（4*4）
        print(
            f"feature corresponsence set: \nshape: {corr_set.shape} \n values: \n{corr_set.T}"
        )

        # transform & visualize
        pcd_reg = o3d.geometry.PointCloud()
        pcd_reg.points = src_pcd.points
        pcd_reg.transform(trans)
        src_pcd.paint_uniform_color([1, 0, 0])
        tgt_pcd.paint_uniform_color([0, 1, 0])
        pcd_reg.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([src_pcd, tgt_pcd, pcd_reg])


if __name__ == "__main__":
    # names = ["bunny", "room", "temple"]
    names = ["bunny"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    for name in names:
        pointCloudRegistration(f"{prefix}/{name}-pcd", name)
