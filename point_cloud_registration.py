import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from solve import *
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

    return key_points_fpfh.T


# global registration: 为icp计算初始值
def globalReg(
    src: np.ndarray,
    tgt: np.ndarray,
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    hyperparams,
) -> tuple[np.ndarray, np.ndarray]:
    # 根据特征值找对应点, 找特征值最接近的val_num个对应点
    val_num = src.shape[0] // hyperparams["ratio"]
    corrIdx = np.asarray(find_nn(src_feat, tgt_feat, val_num))
    src_corr = src[corrIdx[:, 0].T]
    tgt_corr = tgt[corrIdx[:, 1].T]

    print(f"src_idx: {corrIdx[:, 0]}")
    print(f"tgt_idx: {corrIdx[:, 1]}")

    # solve
    # R, t = svdSolver(src_corr, tgt_corr)
    R, t = convexRelaxSolver(src_corr.T, tgt_corr.T)
    return R, t


# fine registration
def fineReg(
    src: np.ndarray,
    tgt: np.ndarray,
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    src_all: np.ndarray,
    tgt_all: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    hyperparams: dict,
    solver: str = "svd",
    isVisual=False,
) -> tuple[np.ndarray, np.ndarray]:
    # 先用feature筛选掉一半的点
    corrIdxFeat = find_nn(src_feat, tgt_feat, src.shape[0] // 3).T
    print(corrIdxFeat)
    # corrIdxFeat = corrIdxFeat[:, : corrIdxFeat.shape[0] // 3]
    src_, tgt_ = src[corrIdxFeat[0]], tgt[corrIdxFeat[1]]

    for _ in range(10):
        # 找对应点
        # val_num = src.shape[0] // 3
        # corrIdx = find_nn(src_, tgt, src_feat, tgt_feat, w1, w2, val_num)
        # corrIdx = find_nn(src_, tgt, val_num)
        corrIdx = find_nn_kdTree(src_, tgt_)
        src_corr, tgt_corr = src_[corrIdx[0].T], tgt_[corrIdx[1].T]
        print(f"corrspond index:\n{corrIdx}")

        # solve
        if solver == "svd":
            R_, t_ = svdSolver(src_corr, tgt_corr)
        else:
            R_, t_ = convexRelaxSolver(src_corr, tgt_corr)

        # update
        R = R_ @ R
        t = R_ @ t + t_
        src_ = (R_ @ src_.T).T + t

        # visualize
        if isVisual:
            reg_pcd = (R @ src_all.T).T + t
            pcd_visualize([src_all, reg_pcd, tgt_all])

    return R, t


def pointCloudRegistration(prefix, name, hyperparams):
    # 1. load data
    pcds, infos = loadData(prefix, name)

    # 2. feature extraction
    # 注意：a.获取的是salient points的feature  b.每个feature是一个行向量
    salient_features = [
        featureExtraction(pcd, info["all_idxs"], hyperparams["fpfh"], False)
        for pcd, info in zip(pcds, infos)
    ]

    # 注意：下面的pcd都是ndarray类型，shape = (pointNum, 3)
    tgt_pcd = pcdToNp(pcds[0])
    tgt_pcd_salient = tgt_pcd[infos[0]["all_idxs"]]
    tgt_salient_feature = salient_features[0]

    reg_res = []

    for i in range(2):
        idx = i + 1
        src_pcd = pcdToNp(pcds[idx])
        src_pcd_salient = src_pcd[infos[idx]["all_idxs"]]
        src_salient_feature = salient_features[idx]

        # 3. global registration
        # 注意: 这里的t是向量，不是矩阵
        print("==================start global registration==================")
        R, t = globalReg(
            src_pcd_salient,
            tgt_pcd_salient,
            src_salient_feature,
            tgt_salient_feature,
            hyperparams["globalReg"],
        )

        # visualization
        golReg_pcd = (R @ src_pcd.T).T + t
        pcd_visualize([src_pcd, golReg_pcd, tgt_pcd])

        # 4. fine registration
        print("==================start local registration==================")
        R, t = fineReg(
            src_pcd_salient,
            tgt_pcd_salient,
            src_salient_feature,
            tgt_salient_feature,
            src_pcd,
            tgt_pcd,
            R,
            t,
            hyperparams["fineReg"],
            "svd",
            True,
        )

    return reg_res


if __name__ == "__main__":
    # names = ["bunny", "room", "temple"]
    names = ["temple"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    hyperparams = [
        {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 40,
                "max_nn_fpfh": 70,
            },
            "globalReg": {"ratio": 3},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 1},
        },
        {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "globalReg": {"ratio": 4},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 20},
        },
        {
            "fpfh": {
                "r_normal": 1,
                "r_fpfh": 1,
                "max_nn_norm": 1000,
                "max_nn_fpfh": 1000,
            },
            "globalReg": {"ratio": 3},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 20},
        },
    ]

    reg_res = []

    for idx, name in enumerate(names):
        res = pointCloudRegistration(f"{prefix}/{name}-pcd", name, hyperparams[idx])
