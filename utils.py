import json

import imageio
import matplotlib.pyplot as plt
import numpy as np
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


def pcdToNp(pointCloud):
    return np.asarray(pointCloud.points)


def npToPcd(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd


def pcd_visualize(point_collections: list[np.ndarray]):
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


def saveAsPly(points: np.ndarray, path):
    pcd = npToPcd(points)
    o3d.io.write_point_cloud(path, pcd)


def saveVisibleResults(pcds: list[np.ndarray], path=None, isVisible=False):
    n = len(pcds)
    labels = ["Registrated", "Target", "Source"]
    colors = ["red", "blue", "green"]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the registered and target point clouds
    for i in range(n):
        ax.scatter(
            pcds[i][:, 0],
            pcds[i][:, 1],
            pcds[i][:, 2],
            c=colors[i],
            label=labels[i],
            s=0.1,
        )

    # Set labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if path is not None:
        plt.savefig(path)
    if isVisible:
        plt.show()


def visualize(point_collections, savePath=None):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    merged_pcd = o3d.geometry.PointCloud()

    for i, points in enumerate(point_collections):
        # create PointCloud object & convert ndarray to points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # set color
        pcd.paint_uniform_color(colors[i])
        merged_pcd += pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(merged_pcd)
    vis.run()
    if savePath != None:
        vis.capture_screen_image(savePath)
    vis.destroy_window()


def visualizeGif(point_collections, savePath=None):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_cloud = o3d.geometry.PointCloud()

    for i, points in enumerate(point_collections):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(colors[i])
        point_cloud += pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    # 每次绕y轴旋转10度，获得旋转矩阵并扩展为4*4的transform矩阵
    rotation_angle = 2
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(rotation_angle)), 0, np.sin(np.radians(rotation_angle))],
            [0, 1, 0],
            [
                -np.sin(np.radians(rotation_angle)),
                0,
                np.cos(np.radians(rotation_angle)),
            ],
        ]
    )
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix

    images = []  # List to store individual frames

    # Perform the rotation directly
    for _ in range(180):  # 36 frames for a full rotation (adjust as needed)
        point_cloud.transform(transform_matrix)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Capture the screen image
        image = np.asarray(vis.capture_screen_float_buffer(), dtype=np.uint8) * 255
        images.append(image)  # Convert to uint8

    if savePath != None:
        imageio.mimsave(savePath, images, fps=45)

    # Destroy the window
    vis.destroy_window()
