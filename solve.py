import copy

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
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


def icp(
    src: np.ndarray,
    tgt: np.ndarray,
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    hyperparams: dict
):
    loss = []

    for _ in tqdm(range(hyperparams['maxIters'])):
        # 找对应关系
        w1 = hyperparams["w"]
        w2 = 1 - w1
        sim_feat = com_sim(src_feat, tgt_feat)
        sim_pos = com_sim(src, src)
        sim = w1 * sim_pos + w2 * sim_feat

        sort_ind = np.zeros(sim.shape)
        for i in range(len(sim)):
            sort_ind[i, :] = np.argsort(sim[i, :])[::-1]

        nk = len(tgt)
        val_num = int(np.floor(nk * 0.2))
        src_id, tgtid = pro(sort_ind[:, 0], val_num)
        src_id = list(map(int, src_id))

        R_, t_ = solve(src_[src_id, :].T, tgt[tgtid, :].T)
        if np.linalg.norm(R_ - R) < 1e-6:
            print("break")
            break
        R = R_ @ R
        t = R_ @ t + t_
        loss.append(com_loss((R @ src_[src_id, :].T).T + t, tgt[tgtid, :]))

    print(loss)
    plt.figure()
    plt.plot(range(len(loss)), loss)
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
