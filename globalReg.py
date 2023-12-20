import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import json

# 读取点云数据
name = "temple"
prefix = f"/Users/riverzhao/Documents/研一/convex optimization/project/code/src/data/{name}-pcd"
point_cloud = o3d.io.read_point_cloud(f"{prefix}/{name}-pcd-1.ply")

# 读取关键点索引数组
json_path = f"{prefix}/points.json"
with open(json_path, "r") as file:
    info = json.load(file)

key_points_indices = info["pcd_1"]["all_idxs"]
key_points = point_cloud.select_by_index(key_points_indices)

# 计算法向量 normal（也是需要在邻域计算）
point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

# 构建 FPFH 特征估计器（只能先计算所有的特征，再提取出salient points的特征）
fpfh_estimator = o3d.pipelines.registration.compute_fpfh_feature(
    point_cloud,
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100),
)

# 获取 FPFH 特征
key_points_fpfh = fpfh_estimator.data[:,key_points_indices]

print(type(key_points_fpfh))
print(key_points_fpfh.shape)

fig, ax = plt.subplots(figsize=(10, 6))

# 对关键点
for i in range(key_points_fpfh.shape[1]):
    ax.plot(key_points_fpfh[:, i], label=f'Key Point {i + 1}')

ax.set_xlabel('Dimension')
ax.set_ylabel('FPFH Value')
ax.legend()
plt.show()


# # 将关键点和 FPFH 特征可视化
# point_cloud.paint_uniform_color([0, 1, 0])
# key_points.paint_uniform_color([1, 0, 0])  # 关键点为红色
# o3d.visualization.draw_geometries([point_cloud, key_points])


