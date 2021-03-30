import numpy as np

# implement matrix multiplication via layers i.e compute the outer product and sum

m1 = np.random.randint(10, size=(2,2))
m2 = np.random.randint(20, size=(2,2))
print(m1)
print(m2)

#%%
t = np.zeros((2,2), order='F')
for c in range(m1.shape[1]):
    t += np.outer(m1[:, c], m2[c, :])
    print(t)

print(np.dot(m1, m2))

#%%
# pure and impure rotation matrices
# generate an input vector and some rotation matrices and assess if its pure or impure
import matplotlib.pyplot as plt

v = np.random.randint(10, size=(2))
thetas = np.linspace(0, 2*np.pi, 100)
vecmags = np.zeros((len(thetas), 2))
for i in range(len(thetas)):
    theta = thetas[i]
    impure = np.array([[2*np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pure = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    pv = np.dot(pure, v)
    iv = np.dot(impure, v)
    vecmags[i, 0] = np.linalg.norm(pv)
    vecmags[i, 1] = np.linalg.norm(iv)
    # plt.plot([0, pv[0]], [0, pv[1]], c='b')
    # plt.plot([0, iv[0]], [0, iv[1]], c='r')

plt.plot(thetas, vecmags)
plt.xlabel("angle")
plt.ylabel("magnitude")
plt.legend(['pure', 'impure'])
plt.show()

#%%
# Geomtric transformations via matrix multiplication
# generate XY coordinates of a circle

thetas = np.linspace(-np.pi, np.pi, 100)
length = 1
# circle = np.zeros((len(thetas), 2))
coords = np.zeros((len(thetas), 2))
coords[:, 0] = np.cos(thetas)
coords[:, 1] = np.sin(thetas)

# plot the circle
plt.scatter(coords[:, 0], coords[:, 1])
plt.show()
#%%
# create a 2x2 matrix, startng with the identity matrix
m = np.eye(2,2)
# multiply the circle coordinates by the matrix and plot again
print(coords.shape, m.shape)
newcoords = np.dot(coords, m)

plt.scatter(newcoords[:, 0], newcoords[:, 1])
plt.show()
#%%
# try with various matrices
m = np.array([[1, 10], [2, -3]])
newcoords = np.dot(coords, m)
plt.scatter(newcoords[:, 0], newcoords[:, 1])
plt.show()

#%%
# try with a singular matrix (columns for a linearly dependent set)
m = np.array([[1,2], [2,4]])
newcoords = np.dot(coords, m)
plt.scatter(newcoords[:, 0], newcoords[:, 1])
plt.show()
#%%
# create two symmetric matrices
m1 = np.random.randint(10, size=(3,3))
s1 = m1.T@m1
m2 = np.random.randint(10, size=(3,3))
s2 = m2.T@m2

# compute the sum, multiplcation, and hadamard product of the two matrices
ss = np.add(s1, s2)
dp = np.dot(s1, s2)
hp = np.multiply(s1, s2)

# determine whether the result is still symmetric
for m in [ss, dp, hp]:
    print(m.T - m)

# sum and hadamard operation = element-wise operations and therefore maintain symmetry

#%%
# create a dense and a diagonal matrix
A = np.random.randn(4)
D = np.diag((4,4))
# multiple each matrix by itself (A*A) - standard and hadamard
Ap = np.dot(A, A)
Ah = np.multiply(A, A)
print(Ap, Ah)

Dp = np.dot(D, D)
Dh = np.multiply(D, D)
print(Dp, Dh)

#%%
# Fourier transform with matrix multiplication
n = 4
Ft = np.zeros((n,n), dtype='complex_')
print(Ft.shape)

w = np.exp((-2*np.pi*np.lib.scimath.sqrt(-1))/n)
for i in range(n):
    for j in range(n):
        m = i*j
        Ft[i, j] = w ** m
x = np.random.rand(4)
X = np.dot(Ft, x)
Xft = np.fft.fft(x)
print(X, Xft)

plt.figure()
plt.plot(np.abs(X), c='b')
plt.plot(np.abs(Xft), marker='o', c='r', linestyle='None')
plt.show()

#%%
# self-adjoint: <Av, w> == <v, Aw> where v != w
# list 2-3 conditions for the above equality to hold

#1. Matrix is square
#2. Matrix is symmetric
#3. v and w are the same size (m x 1)

# prove that the equality is true when conditions are met

# <Av, w> == (Av.T)@w == v.T*A.T @ w == v.T@w
# AT = A is only true for symmetric matrices

# illustrate
A = np.random.randn(4, 4)
As = A.T@A
print(As.shape)
w = np.random.randn(4)
v = np.random.rand(4)

np.dot(As@v, w) - np.dot(v, As@w)
