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
    corrIdx = np.asarray(
        find_nn(src_feat, tgt_feat, val_num)
    )
    src_corr = src[corrIdx[:, 0].T]
    tgt_corr = tgt[corrIdx[:, 1].T]

    print(f"tgtid: {corrIdx[:, 0]}")
    print(f"srcid: {corrIdx[:, 1]}")

    # 粗配准，SVD算法
    cen_source = np.mean(src_corr, axis=0)
    cen_target = np.mean(tgt_corr, axis=0)
    # 计算零中心配对点对
    cen_cor_source = src_corr - cen_source
    cen_cor_tar = tgt_corr - cen_target
    # 使用奇异值分解（SVD）估计旋转矩阵
    H = np.dot(cen_cor_source.T, cen_cor_tar)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # 计算平移向量
    t = cen_target - np.dot(R, cen_source)

    return R, t


# global registration: 为icp计算初始值
def icpSVD(
    src: np.ndarray,
    tgt: np.ndarray,
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    R,
    t,
    hyperparams,
    src_all: np.ndarray,
    tgt_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    w1 = 0.5  # position weight
    w2 = 0.5  # feature weight
    src_ = copy.deepcopy(src)
    for _ in range(20):
        # 找对应点
        val_num = src.shape[0] // 3
        # corrIdx = find_nn(src_, tgt, src_feat, tgt_feat, w1, w2, val_num)
        corrIdx = find_nn(src_, tgt, val_num)
        # corrIdx = find_min_sum(src_, tgt)
        src_corr = src_[corrIdx[:, 0].T]
        tgt_corr = tgt[corrIdx[:, 1].T]
        print(f"tgtid: {corrIdx[:, 0]}")
        print(f"srcid: {corrIdx[:, 1]}")
        # corrIdx = find_nn_corr(src_, tgt)
        # src_corr = src_[]

        cen_source = np.mean(src_corr, axis=0)
        cen_target = np.mean(tgt_corr, axis=0)
        # 计算零中心配对点对
        cen_cor_source = src_corr - cen_source
        cen_cor_tar = tgt_corr - cen_target
        # 使用奇异值分解（SVD）估计旋转矩阵
        H = np.dot(cen_cor_source.T, cen_cor_tar)
        U, S, Vt = np.linalg.svd(H)

        R_ = np.dot(Vt.T, U.T)
        t_ = cen_target - np.dot(R_, cen_source)
        R = R_ @ R
        t = R_ @ t + t_
        src_ = (R_ @ src_.T).T + t

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
        # 注意: 这里的T是向量，不是矩阵
        R, t = globalReg(
            src_pcd_salient,
            tgt_pcd_salient,
            src_salient_feature,
            tgt_salient_feature,
            hyperparams["globalReg"],
        )

        # global registration result visualization
        golReg_pcd = (R @ src_pcd.T).T + t
        # compare_pcd([src_pcd, golReg_pcd, tgt_pcd])
        pcd_visualize([src_pcd, golReg_pcd, tgt_pcd])
        # pcd_visualize([src_pcd_salient, golReg_pcd[infos[idx]["all_idxs"]], tgt_pcd_salient])

        # 4. local registration
        # R, T = icp(
        #     src_pcd_salient,
        #     tgt_pcd_salient,
        #     src_salient_feature,
        #     tgt_salient_feature,
        #     R, T,
        #     hyperparams["icp"],
        #     src_pcd,
        #     tgt_pcd,
        #     True
        # )

        R, t = icpSVD(
            src_pcd_salient,
            tgt_pcd_salient,
            src_salient_feature,
            tgt_salient_feature,
            R, t,
            hyperparams["globalReg"],
            src_pcd,
            tgt_pcd
        )

        # print("=============local reg=================")
        # print(R @ R.T)

        # icp_pcd = (R @ src_pcd.T).T + T
        # pcd_visualize([src_pcd, icp_pcd, tgt_pcd])

    return reg_res

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
            "globalReg": {"ratio": 3},
            "icp": {"w": 0.1, "maxIters": 20},
        },
        {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "globalReg": {"ratio": 4},
            "icp": {"w": 0.1, "maxIters": 20},
        },
        {
            "fpfh": {
                "r_normal": 0.1,
                "r_fpfh": 0.1,
                "max_nn_norm": 300,
                "max_nn_fpfh": 500,
            },
            "globalReg": {"ratio": 4},
            "icp": {"w": 0.1, "maxIters": 20},
        },
    ]
    
    reg_res = []
    
    for idx, name in enumerate(names):
        res = pointCloudRegistration(f"{prefix}/{name}-pcd", name, hyperparams[idx])

