import copy

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import *


# class ConvexRelaxationSolver:
# X是src，Y是tgt
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
    R: np.ndarray,
    t: np.ndarray,
    hyperparams: dict,
):
    loss = []
    w = 1  # feature的权重
    s = w / 10 / hyperparams["maxIters"]
    sim_feat = com_sim(src_feat, tgt_feat)
    w1 = 0
    w2 = 1

    for _ in tqdm(range(hyperparams["maxIters"])):
        src_ = np.transpose(R @ copy.deepcopy(src).T) + t

        # 找对应关系
        # sim_pos = com_sim(src_, tgt)
        # sim = (1 - w) * sim_pos + w * sim_feat

        # # w = w - s  # 更新权重，随着迭代的进行，feat的权重越来越小

        # sort_ind = np.zeros(sim.shape)
        # for i in range(len(sim)):
        #     sort_ind[i, :] = np.argsort(sim[i, :])

        val_num = src.shape[0] // 4
        # tgt_idx, src_idx = pro(sort_ind[:, 0], val_num)
        # tgt_idx = [int(x) for x in tgt_idx]
        corr = find_nn_posAndFeat(src, tgt, src_feat, tgt_feat, w1, w2, val_num)
        src_idx = corr[:, 0].T
        tgt_idx = corr[:, 1].T


        R_, t_ = solve(src_[src_idx, :].T, tgt[tgt_idx, :].T)
        if np.linalg.norm(R_ - R) < 1e-6:
            print("early stop, the problem has already converged")
            break
        R = R_ @ R
        t = R_ @ t + t_
        loss.append(com_loss((R @ src_[src_idx, :].T).T + t, tgt[tgt_idx, :]))

    # print(loss)
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.show()

    return R, t


def com_loss(A, B):
    d = np.linalg.norm(A - B, axis=1)  # 计算每一行的距离
    sum_d = np.sum(d)
    return sum_d / len(A)
