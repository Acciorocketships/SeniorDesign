import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial import ConvexHull 
import numpy as np

def graph(points, f = None):
    
    if f is not None:
        points = []
        with open(f, 'r') as doc:
            line = doc.readline()
            while line:
                ls = line.split(',')
                ls[2] = ls[2].strip('\n')
                points.append(ls)
                line = doc.readline()

    fig = plt.figure()
    ax = Axes3D(fig)

    for i in points:
        ax.scatter(float(i[0]), float(i[1]), float(i[2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    graph([], input())
    