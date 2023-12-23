import cvxpy as cp
import numpy as np
from tqdm import tqdm
import copy
import  matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# class ConvexRelaxationSolver:
def solve(X, Y):
    r = cp.Variable((3, 3))
    t = cp.Variable(3)
    C = cp.bmat([
        [1 + r[0][0] + r[1][1] + r[2][2], r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]],
        [r[2][1] - r[1][2], 1 + r[0][0] - r[1][1] - r[2][2], r[1][0] + r[0][1], r[0][2] + r[2][0]],
        [r[0][2] - r[2][0], r[1][0] + r[0][1], 1 - r[0][0] + r[1][1] - r[2][2], r[2][1] + r[1][2]],
        [r[1][0] - r[0][1], r[0][2] + r[2][0], r[2][1] + r[1][2], 1 - r[0][0] - r[1][1] + r[2][2]]
    ])
    constraints = [C >> 0]
    prob = cp.Problem(
        cp.Minimize(cp.norm((r @ X + cp.vstack([t for _ in range(X.shape[1])]).T - Y), p='fro')),
        constraints
    )
    opt = prob.solve(solver='SCS', verbose=False)
    r = r.value
    t = t.value
    if np.linalg.norm(r @ r.T - np.eye(3)) > 1e-3:
        u, s, vh = np.linalg.svd(r)
        r = u @ vh
    return r, t

def icp(tgt, pcd,  pcd_point, tgt_point, iters=10):

    R, t = np.eye(3), np.zeros(3)
    for _ in tqdm(range(iters)):
        pcd_ = (copy.deepcopy(pcd) @ R )+ t
        corr = find_n(tgt, pcd_)  # Find correspondence
        R_, t_ = solve(pcd_[corr,:].T, tgt.T)       # Align
        if (np.linalg.norm(R_-R) < 1e-6):
            break
        R = R_ @ R
        t = R_ @ t + t_
        re = (pcd_point @ R) + t
        com_loss((pcd_[corr,:] @ R) + t, tgt)
        compare_pcd([tgt_point, re], labels=['Original', 'Recovered'], path=str(_))

    return R, t


def compare_pcd(pcds, labels=None, path=None):
    if labels is None:
        labels = ['Original', 'Corrupted', 'Recovered']

    dpi = 80
    fig = plt.figure(figsize=(1440/dpi, 720/dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    ax.set_proj_type('persp')

    for points, label in zip(pcds, labels):
        ax.scatter(points[:, 0], points[:, 2], points[:, 1],
                marker='.', alpha=0.5, edgecolors='none', label=label)

    plt.legend()

    if path is not None:
        if path is not None: plt.savefig(f"./imgs/{path}.png")
    # plt.show()

    # plt.clf()


def find_n(A, B):  # 找最近邻点的索引
    distances = cdist(A, B)
    min_indices = np.argmin(distances, axis=1)
    return min_indices

def com_loss(A,B):
    d = np.linalg.norm(A - B, axis=1)  # 计算每一行的距离
    sum_d = np.sum(d)
    print("loss:", sum_d)

