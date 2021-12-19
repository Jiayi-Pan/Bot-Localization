import numpy as np

# motion model
A = np.eye(2)
B = np.eye(2)

# sensor model
C = np.eye(2)

# motion noise covariance
R = np.array([[2.50696845e-03, 1.79957758e-05],
              [1.79957758e-05, 2.51063277e-03]])

# sensor noise covariance
Q = np.array([[0.04869528, -0.0058636],
              [-0.0058636, 1.01216104]])