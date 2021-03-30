import numpy as np
import matplotlib.pyplot as plt

#%%
# create a 10x10 matrix with rank = 4 using matrix multiplication

m = np.random.randn(10, 4)
n = np.random.randn(4, 10)

t = m@n
print(t.shape)
np.linalg.matrix_rank(t)

#%%
# generalise the procedure to create any mxn matrix with rank r

def maxrank(m, n, r):
    A = np.random.randn(m, r)
    B = np.random.randn(r, n)
    return A@B

c = maxrank(22, 34, 6)
print(c.shape)
print(np.linalg.matrix_rank(c))
#%%
# test whether matrix rank is invariant to scalar multiplication
# create a full rank and a reduced rank matrix

F = np.random.randn(3,3)
R = maxrank(3,10,2)
s = 9
sF = np.multiply(s, F)
sR = np.multiply(s, R)
print(F.shape)
print(R.shape)
print(sF.shape)
print(sR.shape)

#%%
# print ranks of the matrices and the scaled matrices

print(np.linalg.matrix_rank(F))
print(np.linalg.matrix_rank(R))
print(np.linalg.matrix_rank(sF))
print(np.linalg.matrix_rank(sR))

#%%
# check rank(s * F) == s*rank(F) (also a test for linear operation)
sF - s*np.linalg.matrix_rank(F)

#%%
# determine whether a vector is in the span of a set

v = np.array([[1,2,3,4]])
s = np.array([[4,3,6,2], [0,4,0,1]])
t = np.array([[1,2,2,2], [0,0,1,2]])

print(np.linalg.matrix_rank(s))
print(np.linalg.matrix_rank(t))

# n = s + v
n = np.append(s, v, axis=0)
print(n)
print(np.linalg.matrix_rank(n))

m = np.append(t, v, axis=0)
print(m)
print(np.linalg.matrix_rank(m))
#%%


#%%
v = np.random.randn(4)

s = np.random.randn(4,2)

print(np.linalg.matrix_rank(v))
print(np.linalg.matrix_rank(s))
