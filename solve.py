import cvxpy as cp
import numpy as np

from utils import *


# Convex Relaxation Solver
# X是src，Y是tgt
def convexRelaxSolver(X, Y):
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


def svdSolver(src, tgt):
    # 粗配准，SVD算法
    cen_source = np.mean(src, axis=0)
    cen_target = np.mean(tgt, axis=0)
    # 计算零中心配对点对
    cen_cor_source = src - cen_source
    cen_cor_tar = tgt - cen_target
    # 使用奇异值分解（SVD）估计旋转矩阵
    H = np.dot(cen_cor_source.T, cen_cor_tar)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # 计算平移向量
    t = cen_target - np.dot(R, cen_source)

    return R, t
