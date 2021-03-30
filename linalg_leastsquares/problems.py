import numpy as np
import matplotlib.pyplot as plt

# implement linear least squares using QR decomposition
# QR: beta = inv(R.T@R) @ R.T @ Q.T @ y
# mine: beta = inv(R)@Q.T@y

X = np.random.randn(5,5)
y = np.random.randn(5, 1)

Q, R = np.linalg.qr(X)

beta = np.linalg.inv(R.T@R)@R.T@Q.T@y

print(beta)
# compare QR to left inverse method

beta2 = np.linalg.inv(X.T@X)@X.T@y
print(beta2)

beta3 = np.linalg.inv(R)@Q.T@y
print(beta3)
