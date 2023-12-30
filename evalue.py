import subprocess

cmd = []
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/bunny-pcd/bunny-pcd-1.ply  -p2 ./result/bunny-registration/pcd-2-registrated.ply")
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/bunny-pcd/bunny-pcd-1.ply  -p2 ./result/bunny-registration/pcd-3-registrated.ply")
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/room-pcd/room-pcd-1.ply  -p2 ./result/room-registration/pcd-2-registrated.ply")
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/room-pcd/room-pcd-1.ply  -p2 ./result/room-registration/pcd-3-registrated.ply")
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/temple-pcd/temple-pcd-1.ply  -p2 ./result/temple-registration/pcd-2-registrated.ply")
cmd.append("python ./data/metrics.py -c ./data/config.ini -p1 ./data/temple-pcd/temple-pcd-1.ply  -p2 ./result/temple-registration/pcd-3-registrated.ply")

for i in range(len(cmd)):
    subprocess.run(cmd[i])



