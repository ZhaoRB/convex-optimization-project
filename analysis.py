import numpy as np

from utils import *


# 使用FPFH算法计算点云中指定的salient points的特征
def featureExtraction(pointCloud, salientPointsIndex, hyperparams) -> np.ndarray:
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

    return key_points_fpfh.T


def findCorrFromGT(prefix, name, hyperparams):
    # 读入数据
    pcds, infos = loadData(prefix, name)
    # 计算特征
    salient_features = [
        featureExtraction(pcd, info["all_idxs"], hyperparams["fpfh"])
        for pcd, info in zip(pcds, infos)
    ]

    tgt_pcd = pcdToNp(pcds[0])
    tgt_pcd_salient = tgt_pcd[infos[0]["all_idxs"]]
    tgt_salient_feature = salient_features[0]

    for i in range(2):
        idx = i + 1

        src_pcd = pcdToNp(pcds[idx])
        src_pcd_salient = src_pcd[infos[idx]["all_idxs"]]
        src_salient_feature = salient_features[idx]

        R_gt, t_gt = np.asarray(infos[idx]["rotation"]), np.asarray(
            infos[idx]["translation"]
        )

        gt_src_pcd = (R_gt @ src_pcd.T + t_gt).T
        gt_src_pcd_salient = gt_src_pcd[infos[idx]["all_idxs"]]

        # 可视化
        # pcd_visualize([src_pcd, gt_src_pcd, tgt_pcd])
        pcd_visualize([src_pcd_salient, gt_src_pcd_salient, tgt_pcd_salient])

        # 通过 gt 找对应点
        corrIdx = find_nn(gt_src_pcd_salient, tgt_pcd_salient)[:8]
        print(f"src_idx: {corrIdx[:, 0]}")
        print(f"tgt_idx: {corrIdx[:, 1]}")

        # corr = find_nn_kdTree(gt_src_pcd_salient, tgt_pcd_salient)
        # print(f"src_idx: {corr[0]}")
        # print(f"tgt_idx: {corr[1]}")

        # 通过 feature 找对应点
        corrByFeat = find_nn(src_salient_feature, tgt_salient_feature)[:8]
        print(f"src_idx: {corrByFeat[:, 0]}")
        print(f"tgt_idx: {corrByFeat[:, 1]}")


if __name__ == "__main__":
    # names = ["bunny", "room", "temple"]
    names = ["temple"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    hyperparams = {
        "bunny": {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 40,
                "max_nn_fpfh": 80,
            },
            "globalReg": {"ratio": 4},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 1},
        },
        "room": {
            "fpfh": {
                "r_normal": 0.05,
                "r_fpfh": 0.05,
                "max_nn_norm": 30,
                "max_nn_fpfh": 50,
            },
            "globalReg": {"ratio": 4},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 20},
        },
        "temple": {
            "fpfh": {
                "r_normal": 0.5,
                "r_fpfh": 1,
                "max_nn_norm": 500,
                "max_nn_fpfh": 1000,
            },
            "globalReg": {"ratio": 3},
            "fineReg": {"w1": 0.1, "w2": 0.1, "maxIters": 20},
        }
    }

    for idx, name in enumerate(names):
        res = findCorrFromGT(f"{prefix}/{name}-pcd", name, hyperparams[name])