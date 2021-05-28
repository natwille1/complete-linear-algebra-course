import numpy as np
import matplotlib.pyplot as plt

# generate a 2x2 diagonal matrix and compute eigenvalues

a = np.diag([2,2])
eigval, eigvec = np.linalg.eig(a)

print(eigval)
print(eigvec)


#%%
# expand this to nxn diagonal matrices

m = 5
n = 5

a = np.diag(np.diag(np.random.randn(m,n)))
eigval, eigvec = np.linalg.eig(a)
print(a)
print(eigval)
print(eigvec)

### the eigenvectors of a diagonal matrix are always going to be the identity matrix columns and eigenvalues will be the specific diagonal values as they just scale the identity matrix!

#%%
# triangular matrices (upper, lower)

a = np.triu(np.random.randn(3,3))
eigval, eigvec = np.linalg.eig(a)
print(a.shape, eigval.shape, eigvec.shape)
print(a)
print(eigval)
print(eigvec)

### the eigenvalues are the values on the diagonal of the matrix
### the rest of the values are cancelled out by the zero off-diagonal elements that simplify the calculation of the determinant of the matrix when solving the characteristic equation
### the first eigenvector will just be the 1st column of the identity matrix scaled by an eigenvalue that is the value of the first column in the matrix


#%%
a = np.tril(np.random.randn(2,2))
eigval, eigvec = np.linalg.eig(a)
print(a.shape, eigval.shape, eigvec.shape)
print(a)
print(eigval)
print(eigvec)

#%%
# generate 40x40 random matrices
# extract their eigenvalues
# make a plot

a = np.random.randn(40, 40)
eigvals, eigvecs = np.linalg.eig(a)
plt.scatter(np.arange(0, len(eigvals)), eigvals)
# plt.xticks(np.arange(0, len(eigvals)))
plt.show()

#%%
# repeat many times for different random matrices and plot all the eigenvalues together
for i in range(200):
    a = np.random.randn(40,40)
    eigvals, eigenvecs = np.linalg.eig(a)
    plt.scatter(np.arange(0, len(eigvals)), eigvals)
plt.show()


#%%
# prove that eigendecomposition of random matrices and their powers corresponds to applying the same power to the lambda eigenvals matrix and that the eigenvectors are the same

a = np.random.randn(3,3)
b = np.random.randn(3,3)

A = a.T@a # make matrix symmetric so eigenvals are real-valued
B = b.T@b

evals, evecs = np.linalg.eig(A-B)

cvals, cvecs = np.linalg.eig( A@A - A@B - B@A + B@B)

print(np.round(evals, 3))
print(np.round(cvals, 3))

print(np.round(evals**2, 3))
# sort the eigenvals

sidx1 = np.argsort(abs(evals))
sidx2 = np.argsort(cvals)

print(evals[sidx1]**2)
print(cvals[sidx2])

plt.imshow(np.diag(evals**2))
plt.show()
plt.imshow(np.diag(cvals))
np.diag(cvals)
plt.show()


print(evecs[:, sidx1])
print(cvecs[:, sidx2])
print(evecs[:, sidx1] - cvecs[:, sidx2])

# the eigenvectors sometimes don't match because of the sign ambiguity but actually they line on the same subspace

#%%
# symmetric matrices have orthogonal eigenvectors

a = np.random.randn(2,2)
A = a.T@a

evals, evecs = np.linalg.eig(A)
print(A)
print(evecs)

plt.plot(np.array([0, evecs[0][0]]), np.array([0, evecs[0][1]]))
plt.plot(np.array([0, evecs[1][0]]), np.array([0, evecs[1][1]]))
plt.show()

# if evecs are orthogonal, then they're transpose == their inverse so evecs @ evecs.T == identity matrix because the dot product between cols = 0 and the dot product between self-columns = 1 (assuming they are unit normalised)

print(evecs.T@evecs)
plt.imshow(evecs.T@evecs)
plt.show()


#%%
# if matrices are not symmetric

A = np.random.randn(4,4)
evals, evecs = np.linalg.eig(A)

print(evecs)

print(evecs.T@evecs)

plt.imshow(evecs.T@evecs)

#%%
# create mxm symmetric matrix
a = np.random.randn(3,3)
A = a.T@a

# get eigendecomp
evals, evecs = np.linalg.eig(A)

# show norm of outer prod of v_i (corresponds to an eigenvector of mxm matrix) with itself

v1 = evecs[0]
# v1 is now a 1-d np array with no column/vector orientation
# that means v1.T@v1 == v1@v1.T because np doesn't distinguish between this as a column or vector array
# therefore either use the np.outer function to get the outer product (A@A.T)
# or reshape the array to be 2d
v1out = np.outer(v1, v1)
print(v1)
print(v1out)
print(np.linalg.norm(v1out))
print(v1@v1.T) # this will give a single value
print(evals[0]*np.outer(v1,v1))

v1re = np.reshape(v1, (3, 1))
print(v1re.shape)
print(v1re@v1re.T)
print(np.allclose(v1re@v1re.T, np.outer(v1, v1)))
# create 1 layer of eigenlayers matrix A as lambda @ v_i @ v_i transpose (symmetric matrix) and compute its norm
#%%
print(evals[0])
Ai = evals[0]*np.outer(v1,v1)
At = v1re*evals[0]*v1re.T
print(Ai)
print(At)
print(np.linalg.norm(Ai))

#%%
# reconstruct matrix A by summing over the eigenlayers (outer product)
a = np.random.randn(3,3)
A = a.T@a
evals, evecs = np.linalg.eig(A)
Are = np.zeros(A.shape)
for i in range(3):
    v = np.reshape(evecs[i], (3, 1))
    Are += v * evals[i] * v.T
    print(np.linalg.matrix_rank(Are))

print(Are)
print(A)
#%%
# show that the trace of matrix A == sum of eigenvalues of matrix A

A = np.random.randint(1,10, size=(3,3))
print(A)
Atr = np.trace(A)
evals, evecs = np.linalg.eig(A)
print(Atr, np.sum(evals))

# show that the det(A) == product of the eigenvalues

print(np.linalg.det(A))
print(np.prod(evals))

#%%
# repeat above for reduced-rank matrices
A[:, 1] = A[:, 0]
print(A, np.linalg.matrix_rank(A))
Atr = np.trace(A)
evals, evecs = np.linalg.eig(A)
print(evals)# first eval is zero
print(Atr, np.sum(evals))

# show that the det(A) == product of the eigenvalues
print(np.linalg.det(A))
print(np.prod(evals))# product of eigenvals when one is zero is zero
print()
