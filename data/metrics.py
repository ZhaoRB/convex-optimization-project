""" 
Point cloud metrics:
- Chamfer distance
"""
import configargparse
import point_cloud_utils as pcu

def get_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",        is_config_file = True, help='Config file path')
    parser.add_argument("-p1", "--pcd_1",        required = True, help = "Path to the first point clouds", type = str)
    parser.add_argument("-p2", "--pcd_2",        required = True, help = "Path to the second point clouds", type = str)
    return parser.parse_args()

if __name__ == "__main__":
    opts = get_args()
    print(f"Base point cloud loaded from '{opts.pcd_1}'")
    print(f"Registered point cloud loaded from '{opts.pcd_2}'")
    p1 = pcu.load_mesh_v(opts.pcd_1)
    p2 = pcu.load_mesh_v(opts.pcd_2)
    cd = pcu.chamfer_distance(p1, p2)
    print(f"CD:  {cd:.5f}")