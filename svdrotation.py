import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [16,8]
plt.rcParams.update({'font.size': 18})

theta = np.array([np.pi/15, -np.pi/9, -np.pi/20]) #15 for x direction,9 for y and 20 for z 
Sigma = np.diag([3,1,0.5]) #scale x, then y, then z  

#rotation about x-axis
Rx = np.array([[1,0,0],
              [0,np.cos(theta[0]),-np.sin(theta[0])],
              [0,np.sin(theta[0]),np.cos(theta[0])]])
# 3d notation for rotation matrix is:
#Rx = [[1,0,0],
      #[0,cos(A), -sin(A)],
      #[0, sin(A), cos(A)]]

#rotation about y-axis
Ry = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                [0,1,0],
                [-np.sin(theta[1]),0,np.cos(theta[1])]])

#rotation about z-axis
Rz = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                np.sin(theta[2]),np.cos(theta[2]),0,
                [0,0,1]])

#rotate and scale
X = Rz @ Ry @ Rx @ Sigma

#plot sphere
fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')
u = np.linspace(-np.pi,np.pi,100)
v = np.linspace(0,np.pi,100)
x = np.outer(np.cos(u),np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)),np.cos(v))


