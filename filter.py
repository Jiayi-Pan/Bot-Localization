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
    isPermissible = False
    for i in range(len(x_limits)):
        if x < x_limits[i][0] or x > x_limits[i][1]:
            continue
        if y < y_limits[i][0] or y > y_limits[i][1]:
            continue
        isPermissible = True
    return not isPermissible

def ParticleFilter(M, mu, u, z, particles, w):
    # Input: current state estimate mu, control u, measurement
    np.random.seed(34)

    Sigma = np.array([[0.18, 0.0], [0.0, 0.18]])
    # execute control u
    particles = (A @ particles.T + (B @ u).reshape(-1, 1)).T

    for particle_id, particle in enumerate(particles):
        particle_sampled = particle.copy()
        particle_raw = particle.copy()
        particle_sampled[0] = particle[0]
        
        particle_sampled += (B @ u).reshape(-1)
        if checkCollision(particle_sampled):
            particle_sampled_variance = particle_raw.copy()
        else:
        # particle_sampled[0] + np.random.normal(0, 1)
        # particle_sampled[1] = particle[1] + np.random.normal(0, 1)
        # particle[1] += np.random.normal(0, 1)
            particle_sampled_variance = np.random.multivariate_normal(particle_sampled, Sigma)
            while checkCollision(particle_sampled_variance):
                particle_sampled_variance = np.random.multivariate_normal(particle_sampled, Sigma)
                # particle_sampled[0] = particle[0] + np.random.normal(0, 1)
                # particle_sampled[1] = particle[1] + np.random.normal(0, 1)
                # particle[0] += np.random.normal(0, 1)
                # particle[1] += np.random.normal(0, 1)
                # particle_sampled = np.random.multivariate_normal(particles[particle_id], Sigma)
        particles[particle_id] = particle_sampled_variance
    
    # print(w)
    # input()
    # update weights
    for j in range(M):
        w[j] = (1./(2*np.pi*np.linalg.det(Sigma))) * np.exp(-0.5 * (z - C @ particles[j, :]) @ np.linalg.inv(R) @ (z - C @ particles[j, :]))
        # w[j] = 1/np.linalg.norm(z-particles[j, :])
    # resample
    w = w/np.sum(w)
    # print(w)
    # print("weight min =", w.min())
    # input()
    ind = np.random.choice(M, M, p=w)
    particles[:, :] = particles[ind, :]

    # return estimate (mean)
    return particles, w
