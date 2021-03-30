import numpy as np

# test for some random MXN matrices whether s(A+B) = sA + sB
m = 2
n = 3
m1 = np.random.randn(m, n)
m2 = np.random.randn(m, n)
s1 = 25

m3 = s1*(m1+m2)
m4 = s1*m1 + s1*m2
m4 - m3

#%%

# determine the relationship between tr(A) + tr(B) and tr(A+B)
t1 = np.trace(m1) + np.trace(m2)
t2 = np.trace(m1+m2)
t1 - t2

#%%
# determine the relationship between s(tr(A)) and tr(sA)
t1 = s1*np.trace(m1)
t2 = np.trace(s1*m1)

t1-t2


#%%
t = np.random.randint(10, size=(2,3))
print(t)
print(np.trace(t))
