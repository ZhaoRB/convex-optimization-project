import copy

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils import *


# class ConvexRelaxationSolver:
def solve(X, Y):
    r = cp.Variable((3, 3))
    t = cp.Variable(3)
    C = cp.bmat(
        [
            [
                1 + r[0][0] + r[1][1] + r[2][2],
                r[2][1] - r[1][2],
                r[0][2] - r[2][0],
                r[1][0] - r[0][1],
            ],
            [
                r[2][1] - r[1][2],
                1 + r[0][0] - r[1][1] - r[2][2],
                r[1][0] + r[0][1],
                r[0][2] + r[2][0],
            ],
            [
                r[0][2] - r[2][0],
                r[1][0] + r[0][1],
                1 - r[0][0] + r[1][1] - r[2][2],
                r[2][1] + r[1][2],
            ],
            [
                r[1][0] - r[0][1],
                r[0][2] + r[2][0],
                r[2][1] + r[1][2],
                1 - r[0][0] - r[1][1] + r[2][2],
            ],
        ]
    )
    constraints = [C >> 0]
    prob = cp.Problem(
        cp.Minimize(
            cp.norm((r @ X + cp.vstack([t for _ in range(X.shape[1])]).T - Y), p="fro")
        ),
        constraints,
    )
    opt = prob.solve(solver="SCS", verbose=False)
    r = r.value
    t = t.value
    if np.linalg.norm(r @ r.T - np.eye(3)) > 1e-3:
        u, s, vh = np.linalg.svd(r)
        r = u @ vh
    return r, t

'''
hyperparams:
- maxiters
- w1
- w2
'''
def icp(src_sal, tgt_sal, src, tgt, src_feat, tgt_feat, R, T, hyperparams):
    loss = []

    for _ in tqdm(range(iters)):
        pcd_ = (R @ copy.deepcopy(pcd).T).T + t

        re = (R @ pcd_point.T).T + t
        compare_pcd([tgt_point, re], labels=["Original", "Recovered"], path=str(_))

        # 同时考虑feature xyz 找对应关系

        w2 = 1 - w1

        sim_fea = com_sim(tgt_fea, pcd_fea)
        sim_pos = com_sim(tgt, pcd_)
        sim = w1 * (1 - sim_pos) + w2 * (1 - sim_fea)

        sort_ind = np.zeros(sim.shape)
        for i in range(len(sim)):
            sort_ind[i, :] = np.argsort(sim[i, :])[::-1]

        nk = len(tgt)
        # val_num = int(np.floor(nk * 0.25))
        val_num = int(np.floor(nk * 0.2))
        pcd_id, tgtid = pro(sort_ind[:, 0], val_num)
        pcd_id = list(map(int, pcd_id))



        R_, t_ = solve(pcd_[pcd_id, :].T, tgt[tgtid, :].T)
        if np.linalg.norm(R_ - R) < 1e-6:
            print("break")
            break
        R = R_ @ R
        t = R_ @ t + t_

        loss.append(com_loss((R @ pcd_[pcd_id, :].T).T + t, tgt[tgtid, :]))

    print(loss)
    plt.figure()
    plt.plot(range(len(loss)), loss)
    # # plt.title(str(w1))
    plt.show()

    return R, t


def com_loss(A, B):
    d = np.linalg.norm(A - B, axis=1)  # 计算每一行的距离
    sum_d = np.sum(d)
    return sum_d / len(A)


def fnn(Pf, Q):
    kdtree = cKDTree(Q)
    nearest_neighbors = kdtree.query(Pf, k=1)
    return nearest_neighbors[1]
