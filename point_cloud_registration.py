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


def registration(
    src: np.ndarray,
    tgt: np.ndarray,
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    hyperparams,
) -> tuple[np.ndarray, np.ndarray]:
    # 根据特征值找对应点, 找特征值最接近的val_num个对应点
    val_num = tgt.shape[0] // hyperparams["ratio"]
    corrIdx = np.asarray(find_nn(src_feat, tgt_feat, val_num))
    src_corr = src[corrIdx[:, 0].T]
    tgt_corr = tgt[corrIdx[:, 1].T]
    print(f"src_idx: {corrIdx[:, 0]}")
    print(f"tgt_idx: {corrIdx[:, 1]}")

    # 继续筛选对应点
    corrIdxIdx = findCorrSubPcd(src_corr, tgt_corr, hyperparams["threshold"])
    src_corr = src_corr[corrIdxIdx]
    tgt_corr = tgt_corr[corrIdxIdx]
    print(f"src_idx: {corrIdx[:, 0][corrIdxIdx]}")
    print(f"tgt_idx: {corrIdx[:, 1][corrIdxIdx]}")

    # solve
    # R, t = svdSolver(src_corr, tgt_corr)
    R, t = convexRelaxSolver(src_corr.T, tgt_corr.T)
    return R, t


def pointCloudRegistration(prefix, name, hyperparams):
    # 1. load data
    pcds, infos = loadData(f"{prefix}/data/{name}-pcd", name)

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

    for i in range(2):
        idx = i + 1
        src_pcd = pcdToNp(pcds[idx])
        src_pcd_salient = src_pcd[infos[idx]["all_idxs"]]
        src_salient_feature = salient_features[idx]

        # 3. 配准 registration
        # 注意: 这里的t是向量，不是矩阵
        print("==================start registration==================")
        R, t = registration(
            src_pcd_salient,
            tgt_pcd_salient,
            src_salient_feature,
            tgt_salient_feature,
            hyperparams["registration"],
        )

        # visualization
        reg_pcd = (R @ src_pcd.T).T + t
        # pcd_visualize([src_pcd, reg_pcd, tgt_pcd])

        # save results
        saveAsPly(
            reg_pcd, f"{prefix}/result/{name}-registration/pcd-{idx+1}-registrated.ply"
        )
        saveVisibleResults(
            [reg_pcd, tgt_pcd, src_pcd],
            f"{prefix}/result/{name}-registration/pcd-{idx+1}.png",
        )


if __name__ == "__main__":
    names = ["bunny", "room", "temple"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/"
    hyperparams = {
        "bunny": {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 40,
                "max_nn_fpfh": 50,
            },
            "registration": {"ratio": 4, "threshold": 6e-3},
        },
        "room": {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "registration": {"ratio": 4, "threshold": 0.1},
        },
        "temple": {
            "fpfh": {
                "r_normal": 0.5,
                "r_fpfh": 1.2,
                "max_nn_norm": 500,
                "max_nn_fpfh": 1200,
            },
            "registration": {"ratio": 3, "threshold": 5e-2},
        },
    }

    for idx, name in enumerate(names):
        pointCloudRegistration(prefix, name, hyperparams[name])
