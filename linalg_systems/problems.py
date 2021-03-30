import numpy as np
from sympy import *
# Compute row reduced echelon form of matrices of different sizes
# square

A = np.random.randn(4,4)
m = Matrix(A)
m.shape
rA = m.rref()
print(A, rA)

#%%
# rectangular

# wide
B = np.random.randn(4, 10)
b = Matrix(B)
rB = b.rref()
print(B, rB)


# tall
B = np.random.randn(10, 4)
b = Matrix(B)
rB = b.rref()
print(B, rB)
#%%
# linearly dependent columns

C = np.random.randn(4,4)
C[:, 1] = C[:, 2]
rC = Matrix(C).rref()

print(C, rC)

# linearly dependent rows

C = np.random.randn(4,4)
C[1, :] = C[2, :]
rC = Matrix(C).rref()

print(C, rC)
