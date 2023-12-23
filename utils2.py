import json
import open3d as o3d


def loadData(pathPrefix: str, name: str):
    pcd_1 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-1.ply")
    pcd_2 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-2.ply")
    pcd_3 = o3d.io.read_point_cloud(f"{pathPrefix}/{name}-pcd-3.ply")

    json_path = f"{pathPrefix}/points.json"
    with open(json_path, "r") as file:
        info = json.load(file)
    info_1 = info["pcd_1"]
    info_2 = info["pcd_2"]
    info_3 = info["pcd_3"]

    return [pcd_1, pcd_2, pcd_3], [info_1, info_2, info_3]


# visualization
def pcd_visualize(point_collections):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    merged_pcd = o3d.geometry.PointCloud()

    for i, points in enumerate(point_collections):
        # create PointCloud object & convert ndarray to points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # set color
        pcd.paint_uniform_color(colors[i])
        merged_pcd += pcd

    o3d.visualization.draw_geometries([merged_pcd])


def loadSalientInd(pathPrefix: str):

    json_path = f"{pathPrefix}/points.json"
    with open(json_path, "r") as file:
        info = json.load(file)
    info_1 = info["pcd_1"]
    info_2 = info["pcd_2"]
    info_3 = info["pcd_3"]
    p1_ids = info_1['all_idxs']
    p2_ids = info_2['all_idxs']
    p3_ids = info_3['all_idxs']

    return [p1_ids, p2_ids, p3_ids]


