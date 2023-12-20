import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from constants import *
from utils import *

# 使用FPFH算法计算点云中指定的salient points的特征
def featureExtraction(pointCloud, salientPointsIndex) -> np.ndarray:
    pointCloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

    # 构建 FPFH 特征估计器（只能先计算所有的特征，再提取出salient points的特征）
    fpfh_estimator = o3d.pipelines.registration.compute_fpfh_feature(
        pointCloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100),
    )

    # 获取 FPFH 特征
    key_points_fpfh = fpfh_estimator.data[:, salientPointsIndex]

    print(type(key_points_fpfh))
    print(key_points_fpfh.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(key_points_fpfh.shape[1]):
        ax.plot(key_points_fpfh[:, i], label=f"Key Point {i + 1}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("FPFH Value")
    ax.legend()
    plt.show()

    return key_points_fpfh

# 这一步要建立凸优化模型了，计算欧式距离，最小的就是
def featureMatching(src: np.ndarray, tgt: np.ndarray):
    pass



def pointCloudRegistration(prefix, name):
    # 1. load data
    pcds, infos = loadData(prefix, name)

    # 2. feature extraction
    feature = [featureExtraction(pcd, info) for pcd, info in zip(pcds, infos)]

    # 3. feature matching



if __name__ == "__main__":
    names = ["bunny", "room", "temple"]
    prefix = "/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/"
    for name in names:
        pointCloudRegistration(name, prefix)
