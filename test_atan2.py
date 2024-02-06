import math

import matplotlib.pyplot as plt
import numpy


def paint_circle():
    R=100
    x=[]
    y=[]
    for i in range(16):
        x.append(math.cos(i*22.5)*R)
        y.append(math.sin(i*22.5)*R)
    print(x)
    print(y)
    plt.plot(x,y)
    plt.grid()
    plt.show()

def paint_cos():
    R=100
    x=[]
    y=[]
    for i in range(200):
        x.append(i)
        y.append(math.cos(i/10))
    print(x)
    print(y)
    x=numpy.array(x)
    y=numpy.array(y)
    x.sort()

    plt.plot(x, y)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    paint_cos()