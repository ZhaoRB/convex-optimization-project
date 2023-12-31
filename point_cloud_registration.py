import numpy as np
import open3d as o3d

from feature import *
from solve import *
from utils import *


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
        visualize(
            [reg_pcd, tgt_pcd],
            # f"{prefix}/result/{name}-registration/pcd-{idx+1}.png"
        )
        # visualizeGif(
        #     [reg_pcd, tgt_pcd],
        #     # f"{prefix}/result/{name}-registration/pcd-{idx+1}.gif"
        # )

        # save results
        # saveAsPly(
        #     reg_pcd, f"{prefix}/result/{name}-registration/pcd-{idx+1}-registrated.ply"
        # )


if __name__ == "__main__":
    names = ["bunny", "room", "temple"]
    # names = ["bunny"]
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
