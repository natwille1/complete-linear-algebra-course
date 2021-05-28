import numpy as np
import matplotlib.pyplot as plt


# what is the relationship between eig and SVD for a square symmetric matrix?
# create a symmetric matrix (5x5)

a = np.random.randn(5,5)
s = a.T@a


# eig (W, L)

L, W = np.linalg.eig(s)

# eigenvals are not gauranteed to be sorted but singular values are
sidx = np.argsort(L)[::-1] #sort in descending order
L = L[sidx]
W = W[:, sidx]

# svd (U, S, V)

U, S, V = np.linalg.svd(s)

print(f"eigenvals {L}")
print(f"eigenvals sqrt {np.sqrt(L)}")
print(f"singular values {S}")

print(U)
print(W)

# compare U and V and between U and W
fig, ax = plt.subplots(1, 4, figsize=(12,8))
ax[0].imshow(a)
ax[0].set_title("A")
ax[1].imshow(W)
ax[1].set_title("W (eigenvectors)") # ambiguity in the signs of eigenvectors so actually the subspace spanned by W is the same as U
ax[2].imshow(U)
ax[2].set_title("U (left singular vectors)")
ax[3].imshow(V.T)
ax[3].set_title("V (right singular vectors)")
plt.show()

#%%
# create a matrix (3x6)

a = np.random.randn(3, 6)
A = a.T@a

# compute svd and eig of A.T A

U, S, Vt = np.linalg.svd(a)
evals, evecs = np.linalg.eig(A)
sidx = np.argsort(evals)[::-1]
evals = evals[sidx]
evecs = evecs[:, sidx]

# confirm that singular vectors (V) are the same as eigenvectors

print(Vt.T)
print(evecs)
print(Vt.T - evecs)

#%%
# create U using only A, V, and L
print(a.shape, Vt[0, :].shape)
Q = np.zeros((a.shape[0], a.shape[0]))
print(Q.shape)
for i in range(a.shape[0]):
    q = a @ Vt[i, :] / np.sqrt(evals[i])
    Q[:, i] = q

print(Q)

# confirm U == U from formal svd function
print(U)

#%%
# show that SVD identifies the cols and rows of the matrix that contain the most "information"


A = np.random.randn(5,10)

U, E, V = np.linalg.svd(A)
fig, ax = plt.subplots(1, 3, figsize=(12,8))
ax[0].imshow(U)
ax[0].set_title("U (right singular vectors)")
ax[1].imshow(np.diag(E))
ax[1].set_title("E (singular values)")
ax[2].imshow(V.T)
ax[2].set_title("V (left singular values)")
plt.show()
plt.plot(E, 'ks-')
plt.show()


#%%
print(U.shape, E.shape, V.shape)
for i in range(A.shape[0]):
    l1 = E[i] * np.outer(U[:, i], V[i,:])
    plt.imshow(l1)
    plt.title(f"layer {i}")
    plt.show()

#%%
# Generate a matrix such that U*V.T = valid

A = np.random.randn(3, 3) # A has to be symmetric otherwise dimensions of U and V won't match
u, e, v = np.linalg.svd(A)

u@v

#%%
# Compute the norm of U, V, and U*V.T

u_norm = np.linalg.norm(u, 2) # 2 = frobenius norm
v_norm = np.linalg.norm(v, 2)
uv_norm = np.linalg.norm(u@v, 2)
print(f"u norm {u_norm}, v_norm {v_norm}, uv_norm {uv_norm}")

#%%
# Test for the result of UxU.T, V*V.T and U*V.T
plt.imshow(u@u.T)
plt.title("u @ u.t") # identity matrix as U = orthonormal
plt.show()
plt.imshow(v@v.T)
plt.title("v @ v.t") # identity matrix as V/V.T = orthonormal
plt.show()
plt.imshow(u@v.T)
plt.title("u @ v.t")
plt.show()
plt.imshow(A)
plt.title("A")
plt.show()

c = u@v.T
p = c@c.T # this produces the identity matrix because c = orthonormal
print(p)

#%%
# Create a matrix with a specific condition number

def get_mat_cn(cn = 1):
    A = np.random.randn(3,4)
    Q,R = np.linalg.qr(A)
    Q[:, 0] = cn*Q[:, 0]
    return Q

def get_cn(e_mat):
    return np.max(e_mat) / np.min(e_mat)


A = get_mat_cn(42)
u, e, v = np.linalg.svd(A)
cn = get_cn(e)
print(cn)

#%%
# Other way to the do the above is to generate orthonormal matrices of U and V and just multiple these with a constructed E matrix that contains the specified condition number as its highest value
cn = 42
U, _ = np.linalg.qr(np.random.randn(10, 10)) #col space
V, _ = np.linalg.qr(np.random.randn(20, 20)) #row space
e = np.linspace(42,1,10)
E = np.zeros((10,20)) # will have zeros in the remaining columns
for i in range(len(e)):
    E[i, i] = e[i]
A = U@E@V.T
print(np.linalg.cond(A))

#%%
# Why to avoid the inverse

# create a matrix with a known condition number

# calc the explicit inverse

# multiple the original with the inverse

# compute the norm of the difference matrix between calculated I and np.eye

# repeat for matrix sizes between 2-70

# repeat for condition numbers between 10 and 10^12

# show the results of the norms of the difference matrices
def get_mat(dim, cond):
    U, _ = np.linalg.qr(np.random.randn(dim, dim)) #col space
    V, _ = np.linalg.qr(np.random.randn(dim, dim)) #row space
    e = np.linspace(cond,1,dim)
    E = np.zeros((dim,dim)) # will have zeros in the remaining columns
    for i in range(len(e)):
        E[i, i] = e[i]
    A = U@E@V.T
    return A

cond_nums = np.linspace(10, 10**12, 40)
mat_sizes = np.arange(2, 71)
results = np.zeros((len(mat_sizes), len(cond_nums)))

for m, i in enumerate(mat_sizes):
    for c, j in enumerate(cond_nums):
        A = get_mat(i, j)
        Ainv = np.linalg.inv(A)
        Ic = A@Ainv
        diff = abs(Ic - np.eye(i))
        norm_diff = np.linalg.norm(diff, 2)
        results[m,c] = norm_diff


#%%
plt.figure(figsize=(12,8))
# plt.imshow(results)
plt.pcolor(cond_nums, mat_sizes, results)
plt.colorbar()
plt.xlabel("condition number")
plt.ylabel("matrix size")

#%%
# PDF questions
# 2.b)

a = np.array([[0,1,0], [0,0,0]]).T
eigvals, eigvecs = np.linalg.eig(a@a.T)
print(eigvals)
print(eigvecs)

#%%
# 3)
# 1. Generate a 2 × 3 matrix of random numbers.
import numpy as np
import matplotlib.pyplot as plt

a = np.random.randn(2,3)

# 2. Compute its SVD.

u,e,v = np.linalg.svd(a)

# 3. Compute two eigendecompositions using the matrix and its transpose.

evals, evecs = np.linalg.eig(a.T@a)
evals_t, evecs_t = np.linalg.eig(a@a.T)

# 4. Confirm that the eigenvalues and the singular values match for all three decompositions, as
# predicted by the math.

print(f"singular values {e}, \n evals {np.sqrt(evals)}, evals_t {np.sqrt(evals_t)}")

plt.figure(figsize=(12,8))
plt.subplot(141)
plt.imshow(u)
plt.title("U")
plt.subplot(142)
plt.imshow(evecs_t)
plt.title("eig(a@a.T)")
plt.subplot(143)
plt.imshow(v)
plt.title("V")
plt.subplot(144)
plt.imshow(evecs)
plt.title("eig(a.T@a)")
plt.show()

# 5. Plot the eigenvectors and singular vectors in 2D or 3D (as appropriate) to confirm that SVD
# and transpose+eigendecomposition produce the same eigenspaces.
plt.figure(figsize=(12,4))
plt.subplot(141)
plt.plot([0, u[0, 0]], [0, u[1, 0]], label='u_vec1')
plt.plot([0, u[0, 1]], [0, u[1, 1]], label='u_vec2')
plt.title("U")
plt.legend()

plt.subplot(142)
plt.plot([0, evecs_t[0, 0]], [0, evecs_t[1, 0]], label='t_vec1')
plt.plot([0, evecs_t[0, 1]], [0, evecs_t[1, 1]], label='t_vec2')
plt.title("eig(a@a.T)")
plt.legend()

plt.subplot(143)
plt.plot([0, v[0, 0]], [0, v[1, 0]], label='v_vec1')
plt.plot([0, v[0, 1]], [0, v[1, 1]], label='v_vec2')
plt.plot([0, v[0, 2]], [0, v[1, 2]], label='v_vec3')
plt.title("V")
plt.legend()

plt.subplot(144)
plt.plot([0, evecs[0, 0]], [0, evecs[1, 0]], label='eig_vec1')
plt.plot([0, evecs[0, 1]], [0, evecs[1, 1]], label='eig_vec2')
plt.plot([0, evecs[0, 2]], [0, evecs[1, 2]], label='eig_vec3')
plt.title("eig(a.T@a")

plt.legend()
plt.show()
