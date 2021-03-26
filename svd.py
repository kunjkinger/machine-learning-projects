from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16,8] # convert this bpic into 1600 by 800 pixels

A = imread('kunj.jpg')
X = np.mean(A, -1); # converting image in to grayscale

img = plt.imshow(236-X) # to invert the phtoto we use 256-X
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False) # full_matrices means economy svd
#this means not m*n matrix this means m by m matrix
S = np.diag(S) # diagonal matrix

j = 0

for r in (5, 20, 100):
    #construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:] # take the r coloumns of u and S r by r coloumns annd vT for r coloumns
    plt.figure(j+1)
    j += 1
    img = plt.imshow(256-Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()
    
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular  values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('singluar values: commulative sum')
plt.show()
    