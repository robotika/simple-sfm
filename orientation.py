import cv2

class OrientationEstimator:
    'Uses a view graph of keyframes to estimate current orientation.'
    def __init__(self):
        pass
    def add_image(self, img):
        'Returns updated estimate of current orientation.'
        #estimate if keyframe should be added (enough paralax)
        #add keyframe to viewgraph
        pass

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

