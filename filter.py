import numpy as np

#implement the Kalman filter in this function
def KalmanFilter(mu, Sigma, z, u, A, B, C, Q, R):
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

def ParticleFilter():
    #initialize variables
    mu = np.array([[0.], [0.]])
    Sigma = np.array([[1., 0.], [0., 1.]])
    u = np.array([[0.], [0.]])
    z = np.array([[0.], [0.]])
    A = np.array([[1., 0.], [0., 1.]])
    B = np.array([[0.], [0.]])
    C = np.array([[1., 0.], [0., 1.]])
    Q = np.array([[0.1, 0.], [0., 0.1]])
    R = np.array([[0.1, 0.], [0., 0.1]])
    M = 100 #number of particles
    x = np.zeros((2, M))
    w = np.zeros(M)
    #initialize particles
    particles = np.zeros((2, M))
    particles[:,:] = np.random.multivariate_normal(mu.T, Sigma, M)
    #initialize weights
    w[:] = 1./M
    #run the filter
    for i in range(100):
        #get measurement
        z[0] = np.random.normal(0, 1)
        z[1] = np.random.normal(0, 1)
        #update particles
        particles[:,:] = np.random.multivariate_normal(mu.T, Sigma, M)
        #update weights
        for j in range(M):
            x[:,j] = particles[:,j]
            w[j] = w[j] * (1./(2*np.pi*np.linalg.det(Sigma))) * np.exp(-0.5 * (z - C @ x[:,j]) @ np.linalg.inv(R) @ (z - C @ x[:,j]))
        #resample
        w = w/np.sum(w)
        ind = np.random.choice(M, M, p=w)
        particles[:,:] = particles[:,ind]