import open3d as o3d
import numpy as np

VOXEL_GRID_SIZE = 0.01
RADIUS_NORMAL = 20
RADIUS_FEATURE = 50
MAX_SACIA_ITERATIONS = 1000
MIN_CORRESPONDENCE_DIST = 0.01
MAX_CORRESPONDENCE_DIST = 1000

def voxel_filter(cloud, grid_size):
    return cloud.voxel_down_sample(voxel_size=grid_size)

# def get_normals(cloud, radius):
#     return cloud.compute_normals(radius=radius)

def get_normals(cloud, radius):
    # Estimate normals for the point cloud
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Return the estimated normals
    return cloud.normals

def get_features(cloud, normals, radius):
    return o3d.pipelines.registration.compute_fpfh_feature(cloud, normals, radius)

def sac_ia_align(source, target, source_feature, target_feature):
    sac_ia = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, True,
        max_correspondence_distance=MAX_CORRESPONDENCE_DIST,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=MAX_SACIA_ITERATIONS,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(MAX_SACIA_ITERATIONS, 0.999))
    
    return sac_ia.transformation

def view_pair(cloud1, cloud2, cloud1al, cloud2al):
    o3d.visualization.draw_geometries([cloud1, cloud2], window_name="Before Alignment")
    o3d.visualization.draw_geometries([cloud1al, cloud2al], window_name="After Alignment")

def main():
    source = o3d.io.read_point_cloud("./data/bunny/bunny-pcd-1.ply")
    target = o3d.io.read_point_cloud("./data/bunny/bunny-pcd-2.ply")

    # Voxel Downsampling
    source = voxel_filter(source, VOXEL_GRID_SIZE)
    target = voxel_filter(target, VOXEL_GRID_SIZE)

    # Normal Estimation
    source_normals = get_normals(source, RADIUS_NORMAL)
    target_normals = get_normals(target, RADIUS_NORMAL)

    # FPFH Feature Extraction
    source_features = get_features(source, source_normals, RADIUS_FEATURE)
    target_features = get_features(target, target_normals, RADIUS_FEATURE)

    # SAC-IA Registration
    init_transform = sac_ia_align(source, target, source_features, target_features)

    # Transform target cloud
    result = target.transform(init_transform)

    print(init_transform)

    # Visualize
    view_pair(source, target, source, result)

if __name__ == "__main__":
    main()
