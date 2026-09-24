from initialize import ExtendedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1

def f(x, u=0):
    A, k = x[0], x[1]
    A_next = A - dt * k * A
    k_next = k
    return np.array([A_next, k_next])

def F_jacobian(x, u=0):
    A, k = x[0], x[1]

    return np.array([
        [1.0 - dt * k, - dt * A],
        [0.0, 1.0]
    ])

def g(x):
    return np.array([x[0]])

def G_jacobian(x):
    return np.array([[1.0, 0.0]])

np.random.seed(42)
N_steps = 100

x_real = np.array([10.00, 0.3]) #mol/dm^3, 1/s

real_states = []
measurements = []

process_noise_std = 0.01
measurement_noise_std = 0.2

for _ in range(N_steps):

    x_real = f(x_real)
    x_real[0] += np.random.normal(0, process_noise_std)

    z = g(x_real) + np.random.normal(0, measurement_noise_std)

    real_states.append(x_real.copy())
    measurements.append(z)

real_states = np.array(real_states)
measurements = np.array(measurements)

#Initialization

mu_0 = np.array([8.0, 0.05])
Sigma_0 = np.diag([2.0, 0.5])

R = np.diag([1e-4, 1e-5]) #Process noise
Q = np.diag([measurement_noise_std**2]) #Measumenent noise

ekf_of_reaction = ExtendedKalmanFilter(
    f=f,
    F_jacobian = F_jacobian,
    g=g,
    G_jacobian=G_jacobian,
    Q=Q,
    R=R,
    mu=mu_0,
    Sigma=Sigma_0
)

#Estimation cycle

estimated_states = []
estimated_covariances = []

for y in measurements:

    ekf_of_reaction.predict()

    mu, Sigma = ekf_of_reaction.correction(y)

    estimated_states.append(mu.copy())
    estimated_covariances.append(Sigma.copy())

estimated_states = np.array(estimated_states)



