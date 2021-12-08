import numpy as np
import matplotlib.pyplot as plt

a = 6   # Major axis
b = 12   # Minor axis
lr = 0.0001  # Learning rate

# Sample data points
# Input vector 
step = 2000000
xt1 = np.random.uniform(b/(-2),b/2,step) 
xt1 = np.reshape(xt1,(step,1))
t1 = -1*(((a*a/4)*(1 - (4*xt1*xt1)/(b*b)))**(0.5))
t2 = ((a*a/4)*(1 - (4*xt1*xt1)/(b*b)))**(0.5) 
xt2 = np.zeros((step,1))
for i in range(step):
    xt2[i,0] = np.random.uniform(t1[i,0],t2[i,0])
x1 = (xt1+xt2)/(2**(0.5))
x2 = (xt2-xt1)/(2**(0.5))    







# Compute first principal component

x = np.concatenate((x1,x2),axis = 1)
w = np.random.rand(1,2)
y = np.random.rand(step,1)
for i in range(step):

    y[i,0] = np.dot(w,np.transpose(x[i:i+1,0:2]))
    w = w + lr*y[i,0]*(x[i:i+1,0:2]-y[i,0]*w)
    
print(w)

# Compute eigenvalue
y1 = np.random.rand(step,1)
for i in range(step):
    y1[i,0] = np.dot(w,np.transpose(x[i:i+1,0:2]))
v = np.var(y1)
print(v)



