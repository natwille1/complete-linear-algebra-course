import numpy as np
import matplotlib.pyplot as plt

# decompose vector v into orthogonal components

# to be decomposed
w = np.array([2,3])

# reference vector
v = np.array([4, 0])

# compute w parallel to v
# parallel is the orthognal projects of w onto v multiplied by v
par = (np.dot(w, v) / np.dot(v, v)) * v
par

# compute w orthogonal to v
ort = w - par

print(v)
print(w)
print(par)
print(ort)

# confirm results algebraically (sum to w, orthogonal components)
diff = w - (par + ort)
print(diff)

# plot all four vectors
plt.plot(np.array([0, w[0]]), np.array([0, w[1]]), c='b')
plt.plot(np.array([0, v[0]]), np.array([0, v[1]]), c='r')
plt.plot(np.array([0, par[0]]), np.array([0, par[1]]), c='y')
plt.plot(np.array([0, ort[0]]), np.array([0, ort[1]]), c='g')
plt.show()


#%%
# implement the gram-schmidt procedure in code
# start with square matrix, compute Q

def orthognalise(v, w):
    # v = reference vector
    par = (np.dot(w, v) / np.dot(v, v)) * v
    ort = w - par
    return (par, ort)

def gs(A):
    # A = matrix to be orthognalised
    # Q = empty np matrix of same dimensions as A
    Q = np.zeros((A.shape))
    for i in range(A.shape[1]):
        Q[:, i] = A[:, i]
        ref = A[:, i]
        for j in range(i):
            par, ort = orthognalise(Q[:, j], ref)
            Q[:, i] = Q[:, i] - par
        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])
    return Q

A = np.random.randn(4,4)
# Q = np.zeros((4,4))
Q = gs(A)
# check that Q.T Q = I
print(Q)
print(np.round(Q.T@Q),3)
# check that qr() method provides the same answer
q, r =  np.linalg.qr(A)
print(q)


#%%
# check for rectangular matrices
A = np.random.randn(7, 6)
# Q = np.zeros(A.shape)
Qt = gs(A)
print(Qt)
q, r = np.linalg.qr(A, 'complete')
print(Qt.shape)
print(q.shape)

print(np.round(Qt.T@Qt), 3)

#%%
# generate a large (e.g N=100) matrix and invert it using QR and inv() method

A = np.random.randn(100, 100)
q, r = np.linalg.qr(A, 'complete')
print(q)
Ainv = np.linalg.inv(r) @ q.T
print(Ainv)
Ainv2 = np.linalg.inv(A)
print(Ainv2)
plt.imshow(r)
plt.show()
plt.imshow(Ainv)
plt.show()
plt.imshow(Ainv2)
plt.show()

# calcualte correlation to see if they're the same
print(Ainv.flatten().shape)
np.corrcoef(Ainv.flatten(), Ainv2.flatten())

#%%
# show that A.T A == R.T R (from QR)
# generate random matrix A
A = np.random.randn(3,3)
# compute QR of A
q, r = np.linalg.qr(A)
# test
a = A.T @ A
b = r.T @ r
print(A)
print(a)
print(b)
# prove formally


#%%
# questions
a = np.array([-4, 1])
b = np.array([2, 0])
proj = np.dot(a, b) / np.dot(a, a)
print(proj)
plt.plot(b[0],b[1],'ko',label='b')
plt.plot([0, a[0]], [0, a[1]])
# plt.plot([0, b[0]], [0, b[1]])
plt.plot([b[0], proj*a[0]], [b[1], proj*a[1]], 'r--', label='proj')
plt.show()


#%%
a = np.array([2,2])
b = np.array([0, -3])
print(np.dot(a,b)/np.dot(b,b))
print(np.dot(b,b))
print(-2/3)
par = np.dot(a, b) / np.dot(b, b) * b
ort = a - par
print(par, ort)

#%%

A = np.random.randn(5,2)
qs, r = np.linalg.qr(A)
q, r = np.linalg.qr(A, 'complete')
print(qs)
print(q)
iq = q.T@q
iq2 = q@q.T
i = qs.T@qs
i2 = qs@qs.T

print(i)
print(i2)
print(iq)
print(iq2)

#%%

a = np.array([4]).reshape(1, -1)
print(a.shape)
np.linalg.inv(a)
