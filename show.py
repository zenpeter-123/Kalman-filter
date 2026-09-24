from reaction import *
import matplotlib.pyplot as plt

time = np.arange(N_steps) * dt

plt.figure(figsize = (12, 5))

#concentration plot
plt.subplot(1, 2, 1)
plt.plot(time, real_states[:, 0], 'g-', label="real")
plt.plot(time, measurements[:, 0], 'rx', alpha=0.5, label="noisy")
plt.plot(time, estimated_states[:, 0], 'b--', label="EKF-estimate")
plt.title("mol/dm^3")
plt.xlabel("time [s]")
plt.ylabel("concentration")
plt.legend()
plt.grid(True)

#k plot
plt.subplot(1, 2, 2)
plt.plot(time, real_states[:, 1], 'g-', label="real")
plt.plot(time, estimated_states[:, 1], 'b--', label="EKF-estimate")
plt.title("Estimation of k")
plt.xlabel("k [1/s]")
plt.ylabel("time [s]")
plt.legend()
plt.grid(True)

plt.show()
