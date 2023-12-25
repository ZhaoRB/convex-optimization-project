# 暂时用不上的函数
import numpy as np

# 找到矩阵中最小的几个对应点
def findMinInMatrix(matrix, num):
    idx1 = []
    idx2 = []
    visited_row = set()
    visited_col = set()
    for _ in range(num):
        # find minimum
        mini = [0, 0, float("inf")]
        for i in range(matrix.shape[0]):
            if i in visited_row:
                continue
            for j in range(matrix.shape[1]):
                if j in visited_col:
                    continue
                cur = matrix[i][j]
                if cur < mini[2]:
                    mini = [i, j, cur]
        idx1.append(mini[0])
        idx2.append(mini[1])
        visited_row.add(mini[0])
        visited_col.add(mini[1])
        return np.array(idx1), np.array(idx2)