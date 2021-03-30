import numpy as np

# generate 2x2 matrix of integers with linear dependencies
a = np.array([[1,2], [2,4]])
print(np.linalg.det(a))
# calculate the rank
print(np.linalg.matrix_rank(a))

#%%
# generate mxm matrices, impose linear dependencies
m = np.random.randint(100, size=(4,4))
m[:, 0] = m[:, 1]
print(np.linalg.det(m))
# calculate the rank
print(np.linalg.matrix_rank(m))


#%%
# do this for small and large m
m = np.random.randint(100, size=(30, 30))
m[:, 0] = m[:, 1]
m[:, 2] = m[:, 3]
m[:, 4] = m[:, 5]
# print(m[:, :5])
print(np.linalg.det(m))
print(np.linalg.matrix_rank(m))

#%%
#even larger
m = np.random.randint(100, size=(100, 100))
m[:, 0] = m[:, 1]
print(np.linalg.det(m))
print(np.linalg.matrix_rank(m))

# huge determinant even though matrix has a linear dependency - det function not be used to determine if matrix is singular

#%%
# generate 6x6 matrix
m = np.random.randint(100, size=(6,6))
# compute det
print(m)
print(np.linalg.det(m))
#%%
# swap one row - compute det
ms = m[[1,0,2,3,4,5], :]
print(ms)

print(np.linalg.det(ms))
print(np.linalg.det(m))
#%%
# swap two rows - compute det
mss = m[[1,0,3,2,4,5], :]
print(mss)

print(np.linalg.det(mss))
print(np.linalg.det(m))

#%%
# swap two columns
mcs = m[:, [1,0,2,3,4,5]]

print(np.linalg.det(m))
print(np.linalg.det(mcs))


# 2 swaps
mcs = m[:, [1,0,3,2,4,5]]

print(np.linalg.det(m))
print(np.linalg.det(mcs))

#%%
# generate square, random matrix, impose linear dependence
# "shift" matrix by lambda (x identity matrix) in the range of 0-0.1
# compute determinant for shifted matrix
# repeat 1000 times
dets = []
avg = []

lams = list(np.linspace(0, 0.1, 30))
for l in lams:
    for i in range(1000):
        # a = np.random.randint(100, size=(20,20))
        a = np.random.randn(20,20)
        a[:, 0] = a[:, 1]
        s = l * np.eye(20,20)
        a_s = a + s
        dets.append(abs(np.linalg.det(a_s)))
    avg.append(np.mean(dets))


#%%
import matplotlib.pyplot as plt
# plot determinant as function of lambda
plt.plot(lams, avg, '-s')
plt.xlabel("fraction of identity matrix for shifting")
plt.ylabel("determinant")

# determinant increases as we increase the shift towards the identity matrix

#%%
a_s = a + np.eye(20)
print(a[:, 0])
print(np.linalg.det(a))
print(a_s[:, 0])
print(np.linalg.det(a_s))


#%%
# illustrate that det(AB) = det(A) * det(B)
# 1) for random 3x3 matrices

A = np.random.randn(3,3)
B = np.random.randn(3,3)

print(np.dot(A, B))

p1 = np.linalg.det(np.dot(A, B))
p2 = np.linalg.det(A) * np.linalg.det(B)

print(p1, p2)

diff = p1-p2
diff


#%%
# 2) in a loop over random matrix sizes up to 40x40
diffs = []
dets = np.zeros((40,2))
print(dets.shape)
for i in range(40):
    A = np.random.randn(i, i)
    B = np.random.randn(i, i)
    AB = A@B
    dets[i, 0] = np.linalg.det(A) * np.linalg.det(B)
    dets[i, 1] = np.linalg.det(AB)


plt.plot(dets[:,0] - dets[:, 1])
