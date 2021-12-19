import numpy as np
from model import A, B, C, Q, R

#implement the Kalman filter in this function
def KalmanFilter(mu, Sigma, z, u):
    ###YOUR CODE HERE###

    #prediction step    
    mu_pred = A @ mu + B @ u
    Sigma_pred = A @ Sigma @ A.transpose() + R
    
    #correction step
    K = Sigma_pred @ C.T @ np.linalg.inv(C @ Sigma_pred @ C.T + Q)
    mu = mu_pred + K @ (z - C @ mu_pred)
    Sigma = (np.eye(2) - K @ C) @ Sigma_pred

    mu_new = mu; Sigma_new = Sigma #comment this out to use your code
    ###YOUR CODE HERE###
    return mu_new, Sigma_new

def checkCollision(p):
    x_limits = [[0.2, 0.5], [0.4, 0.5]]
    y_limits = [[0.1, 0.2], [0.3, 0.4]]
    x, y = p
    for x_limit in x_limits:
        if x < x_limit[0] or x > x_limit[1]:
            return True
    for y_limit in y_limits:
        if y < y_limit[0] or y > y_limit[1]:
            return True
    return False
    

def ParticleFilter(mu, u, z):
    #initialize variables
    mu = np.array([[0.], [0.]])
    Sigma = np.array([[1., 0.], [0., 1.]])
    
    # u = np.array([[0.], [0.]])
    # z = np.array([[0.], [0.]])

    M = 100 #number of particles
    x = np.zeros((M, 2))
    w = np.zeros(M)
    # initialize particles
    particles = np.zeros((M, 2))
    # for paricle_i in range(M):
    #     particle_sampled = np.random.multivariate_normal(mu, Sigma)
    #     while checkCollision(particle_sampled):
    #         particle_sampled = np.random.multivariate_normal(mu, Sigma)
    #     particles[paricle_i] = particle_sampled
    # particles[:,:] = np.random.multivariate_normal(mu.T, Sigma, M)
    # initialize weights
    w[:] = 1./M
    # run the filter
    for i in range(100):
        # get measurement
        # z[0] = np.random.normal(0, 1)
        # z[1] = np.random.normal(0, 1)
        # sample particles
        for paricle_i in range(M):
            particle_sampled = np.random.multivariate_normal(mu, Sigma)
            while checkCollision(particle_sampled):
                particle_sampled = np.random.multivariate_normal(mu, Sigma)
            particles[paricle_i] = particle_sampled
        
        # particles[:,:] = np.random.multivariate_normal(mu.reshape(-1), Sigma, M)
        
        # update weights
        for j in range(M):
            x[:,j] = particles[:,j]
            w[j] = w[j] * (1./(2*np.pi*np.linalg.det(Sigma))) * np.exp(-0.5 * (z - C @ x[:,j]) @ np.linalg.inv(R) @ (z - C @ x[:,j]))
        # resample
        w = w/np.sum(w)
        ind = np.random.choice(M, M, p=w)
        particles[:,:] = particles[:,ind]
    
    # return estimate (mean)
    return particles.mean(axis=0)