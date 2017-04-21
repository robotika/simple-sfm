import cv2 # features, matching, fundamental matrix
from autograd import numpy as np # gradient

import geometry

class OrientationEstimator:
    'Uses a view graph of keyframes to estimate current orientation.'
    def __init__(self):
        pass
    def add_image(self, img):
        'Returns updated estimate of current orientation.'
        #add img as keyframe to viewgraph
        #prune the graph of old redundant keyframes
        #filter based on triplet error, accumulate over time to support future pruning
        #improve global orientations (fixed number of steps of gradient descent)
        #return orientation of the last keyframe
        return geometry.orientation()

def _detect(img):
    'Returns ORB descriptors of detected features.'
    pass

def _match(img1, img2):
    'Returns index array of matches.'
    #match bruteforce
    pass

def _fund(m1, m2):
    'Returns estimated fundamental matrix'
    pass

def _filter(viewgraph):
    'temporalily deactivate edges based on triplet error'
    pass

def _improve_orientations(viewgraph):
    'gradient descent; returns cumulative error'
    pass

if __name__ == "__main__":
    print('TODO: unit tests')

