import pptk
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
    v = pptk.viewer(points)
    input()

graph(None, f="t.txt")