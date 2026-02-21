import numpy as np
def Relu(z):
    return np.maximum(z,0)


def forward1(x,w1,wout,b1,bout):
    z1 = np.matmul(x,w1)+b1
    A1 = Relu(z1)

    zout = np.matmul(wout,A1)+bout
    Aout = 1/(1+np.exp(-zout))

    return z1,A1,zout,Aout 


