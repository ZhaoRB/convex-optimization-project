# convex-optimization-project

## feature extraction


## global registration

1. 计算质心 + 去中心化（两个点云得到去中心化的坐标）
2. 奇异值分解计算旋转矩阵R（背后的思想是最小二乘法）
3. 根据上面得到的R计算平移向量t