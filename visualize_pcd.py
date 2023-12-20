import open3d as o3d

def visualize_point_clouds(pcd_path1, pcd_path2):
    # Load point clouds from PCD files
    point_cloud1 = o3d.io.read_point_cloud(pcd_path1)
    point_cloud2 = o3d.io.read_point_cloud(pcd_path2)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

# Replace with the actual paths to your PCD files
pcd_file_path1 = "./data/bun1.pcd"
pcd_file_path2 = "./data/bun2.pcd"
visualize_point_clouds(pcd_file_path1, pcd_file_path2)
