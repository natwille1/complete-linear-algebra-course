import numpy as np
import matplotlib.pyplot as plt

# compute and visualise the (normalised) quadratic form for matrix:

a = np.array([[-2, 3], [2, 8]])


n = 40
x = np.linspace(-2,2, n)
qf = np.zeros((n, n))
qfN = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        # xi = np.reshape(np.array([x[i], x[j]]),(-1,1))
        xi = np.transpose([x[i], x[j]])
        qf[i, j] = xi.T@a@xi
        qfN[i, j] = qf[i, j] / (xi.T@xi)

X, Y = np.meshgrid(x, x)
fig, ax = plt.subplots(1,2, figsize=(12,8), subplot_kw={'projection': '3d'})
ax[0].plot_surface(X,Y, qf.T, cmap='viridis')
ax[0].set_title("Raw quad form")
ax[1].plot_surface(X,Y, qfN.T, cmap='viridis')
ax[1].set_title("Normalised quad form")
plt.show()

#%%
# show that eigenvectors point in steepest directions of the quadratic form of a matrix

evals, evecs = np.linalg.eig(a)
sidx = np.argsort(evals)
evals = sorted(evals, reverse=True)
evecs = evecs[:, sidx]
# fig, ax = plt.subplots(1,2, figsize=(12,8), subplot_kw={'projection': '3d'})
plt.imshow(qfN, extent=[-2,2, -2,2])
plt.plot([0, evecs[0, 0]], [0, evecs[1, 0]], label='evec 1')
plt.plot([0, evecs[0, 1]], [0, evecs[1, 1]], label='evec 2')
# ax[0].plot([0, evals[0]*evecs[0, 0]], [0, evals[0]*evecs[1, 0]], label='evec 1')
# ax[0].plot([0, evals[1]*evecs[0, 1]], [0, evals[1]*evecs[1, 1]], label='evec 2')
# ax[1].plot_surface(X,Y, qfN.T, cmap='viridis')
# ax[1].set_title("Normalised quad form")
plt.legend()
plt.show()
