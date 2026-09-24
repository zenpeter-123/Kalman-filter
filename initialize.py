import numpy as np

class KalmanFilter():
    def __init__(self, F, B, G, Q, R, mu, Sigma):
        self.F = F #State transition
        self.B = B #Control
        self.G = G #Measurement
        self.Sigma = Sigma #Estimate error covariance
        self.R = R #Process noise covariance
        self.Q = Q #Observation noise covariance 
        self.mu = mu

    def predict(self, u=0):
        self.mu = self.F @ self.mu + self.B @ u if isinstance(u, np.ndarray) else self.F @ self.mu + self.B * u 
        self.Sigma = self.F @ self.Sigma @ self.F.T + self.R

    def correction(self, y):
        z = y - self.G @ self.mu #Innovation
        S = self.G @ self.Sigma @ self.G.T + self.Q #Innovation covariance
        K = self.Sigma @ self.G.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ z
        self.Sigma = (np.eye(self.Sigma.shape[0]) - K @ self.G) @ self.Sigma

        return self.mu

class ExtendedKalmanFilter():

    def __init__(self, f, F_jacobian, g, G_jacobian, Q, R, mu, Sigma):
        self.f = f
        self.F_jacobian = F_jacobian
        self.g = g
        self.G_jacobian = G_jacobian
        self.Q = Q
        self.R = R
        self.mu = mu
        self.Sigma = Sigma

    def predict(self, u=0):

        F = self.F_jacobian(self.mu, u)

        self.mu = self.f(self.mu, u)

        self.Sigma = F @ self.Sigma @ F.T + self.R

        return self.mu

    def correction(self, y):

        G = self.G_jacobian(self.mu)

        z = y - self.g(self.mu) # Innovation
        S = G @ self.Sigma @ G.T + self.Q #Innovation covariance

        K = self.Sigma @ G.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ z
        self.Sigma = (np.eye(self.Sigma.shape[0]) - K @ G) @ self.Sigma

        return self.mu, self.Sigma
