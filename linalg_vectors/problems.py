import numpy as np

# dot products with matrix columns
# create 2 4x6 matrices of random numbers
# use a for loop to compute the dot product between the corresponding columns of the matrices

m1 = np.random.randn(4,6)
m2 = np.random.randn(4,6)


for i in range(m1.shape[1]):
    dp = np.dot(m1[:, i], m2[:, 1])
    print(dp)


#%%
# is the dot product communutative
# a'*b == b'*a
# 1) generate two 100-element random row vectors and verify communutativity

v1 = np.random.rand(100)
v2 = np.random.rand(100)

d1 = np.dot(v1, v2)
d2 = np.dot(v2, v1)

assert d1 == d2

#%%
# 2) generate two 2-element integer row vectors and repeat

v1 = np.random.randint(100, size=2)
v2 = np.random.randint(100, size=2)

d1 = np.dot(v1, v2)
d2 = np.dot(v2, v1)

assert d1==d2

#%%
# test whether the dot product sign is invariant to scalar multiplication
# generate two 3D vectors
# generate two scalars
# compute dot product between the two vectors
# compute dot product between the scaled vectors

v1 = np.random.randn(3)
v2 = np.random.randn(3)

s1 = np.random.randint(100, size=1)
s2 = np.random.randint(100, size=1)

d1 = np.dot(v1, v2)
sv1 = s1 * v1
sv2 = s2 * v2

d2 = np.dot(sv1, sv2)
print(d1, d2)

#%%
s1 = -10 # changing the sign of the scalar has an affect on the orientation of the vector it is scaling
s2 = 4

d2 = np.dot(s1 * v1, s2 * v2)
print(d1, d2) # therefore sign of the dot product is affected by the scalar

#%%
# create two random int vectors (R4)
# compute the length of the vectors, and the magnitude of the dot product
# normalise the vectors to unit length
# compute the magnitute of the dot product

v1 = np.random.randint(100, size=4)
v2 = np.random.randint(100, size=4)

nv1 = np.linalg.norm(v1)
nv2 = np.linalg.norm(v2)
d1 = np.dot(v1, v2)

print(nv1, nv2, d1)

v1norm = v1 / nv1
v2norm = v2 / nv2

print(np.linalg.norm(v1norm), np.linalg.norm(v2norm))

d2 = np.dot(v1norm, v2norm)
print(d2) # normalised dot product is smaller than 1 and also much smaller than the dot product of unnormalised vectors - this is because the geometric interpretation of the dot product between vectors whose norms = 1 reduces to just 'cos(angle)'  and the cosine is bound between -1 and 1 !! this is cosine similarity !!
