import numpy as np
from model import A, B, C, Q, R

# implement the Kalman filter in this function


def KalmanFilter(mu, u, z, Sigma):

    # prediction step
    mu_pred = A @ mu + B @ u
    Sigma_pred = A @ Sigma @ A.transpose() + R

    # correction step
    K = Sigma_pred @ C.T @ np.linalg.inv(C @ Sigma_pred @ C.T + Q)
    mu = mu_pred + K @ (z - C @ mu_pred)
    Sigma = (np.eye(2) - K @ C) @ Sigma_pred

    mu_new = mu
    Sigma_new = Sigma  # comment this out to use your code
    ###YOUR CODE HERE###
    return mu_new, Sigma_new


def checkCollision(p):
    # Region where permissible
    x_limits = [[-8, -5], [-5, 0], [0, 2], [2, 6], [6, 7], [7, 12], [12, 15]]
    y_limits = [[-8, 6], [-8, -6], [-8, 8], [0, 1], [0, 6], [4, 5], [-8, 6]]
    x, y = p
    assert len(x_limits) == len(y_limits)
    for i in len(x_limits):
        if x < x_limits[i][0] or x > x_limits[i][1]:
            return True
        if y < y_limits[i][0] or y > y_limits[i][1]:
            return True
    return False


def ParticleFilter(M, mu, u, z, particles, w):
    # Input: current state estimate mu, control u, measurement z


    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    # execute control u
    mu_new = A @ mu + B @ u
    
    # update weights
    for j in range(M):
        w[j] = w[j] * (1./(2*np.pi*np.linalg.det(Sigma))) * np.exp(-0.5 *
                                                                   (z - C @ particles[:, j]) @ np.linalg.inv(R) @ (z - C @ particles[:, j]))
    # resample
    w = w/np.sum(w)
    ind = np.random.choice(M, M, p=w)
    particles[:, :] = particles[:, ind]

    # return estimate (mean)
    return particles, w
