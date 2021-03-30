import numpy as np
import matplotlib.pyplot as plt

# implement the mca algorithm
# check against the numpy inv function
# 3 matrices
# M = matrix of dets from submatrices of A
# C = M hadamard with checkerboard of signs - top left = plus sign
# A = C.T * 1/det(A)
m = 3
A = np.random.randn(m,m)
detA = np.linalg.det(A)
print(A)
M = np.zeros((m,m))


#%% M
exclude = np.full((A.shape[1]), False)
for i in range(0, A.shape[1]):
    for j in range(0, A.shape[0]):
        include = np.full((A.shape[0], A.shape[1]), True)
        include[:, i] = exclude
        include[j, :] = exclude
        temp = A[include].reshape(m-1,m-1)
        M[i, j] = np.linalg.det(temp)


#%% C

# C = np.array([[(i+j)%2 for i in range(m)] for j in range(m)])

# C = np.array([[-i, j] for i in range(m)] for j in range(m)])
c = np.ones((m, m))
print(c)
for i in range(m):
    for j in range(m):
        t = i+j
        if t % 2 > 0:
            c[i, j] = -1*c[i, j]

print(c)
print(M)
C = np.multiply(M, c)
C

#%% A

Ainv = C.T * 1/detA
test = Ainv - np.linalg.inv(A)
test

#%%
# create diagonal matrices - start with 2x2 with integers
A = np.diag((2,2))

# compute inverses
Ainv = np.linalg.inv(A)
print(A)
print(Ainv)

#%%
# loop for different matrix values
for i in range(1, 20, 2):
    A = np.diag((i,i))
    print(A)
    Ainv = np.linalg.inv(A)
    plt.imshow(Ainv)
    plt.colorbar()
    plt.clim(1,-1)
    plt.show()


#%%
# loop for matrices of different sizes
for i in range(2, 20, 2):
    A = np.random.randint(0, 100, size=(i, i))
    print(A.shape)
    Ad = np.diag(np.diag(A))
    Adinv = np.linalg.inv(Ad)
    # print(Adinv)
    plt.imshow(Adinv)
    plt.colorbar()
    plt.show()


#%%

# ..think - what conditions on the diagonal matrices for invertibility?
# a column in the matrix cannot be full of zeros -- singular i.e every diagnoal element must be non-zero.
t = np.diagflat([[0, 1, 8]])
print(t)
tinv = np.linalg.inv(t)
plt.imshow(tinv)
plt.colorbar()
plt.show()


#%%
# prove that pseudoinverse is the same to the "real" inverse for invertible matrices

A = np.random.randn(3,3)
print(A)

Ainv = np.linalg.inv(A)
Apinv = np.linalg.pinv(A)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(Ainv)
ax[1].imshow(Apinv)
plt.show()
