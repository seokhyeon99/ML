import numpy as np
def nn(x):
    w = np.array([0.2, -0.1, 0.3])
    b = 0
    v = np.dot(x,w) + b
    if v>0:
        return v
    else:
        return '0'

x = np.array([[0.3, 0.1, 0.8],[0.5,0.6,0.3],[0.1,0.2,0.1],[0.8,0.7,0.7],[0.5,0.5,0.6]])
type(x[1,:])


for i in range(5):
    print(nn(x[i,:]))

nn([1,5,0])

