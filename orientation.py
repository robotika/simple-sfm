import cv2 # features, matching, fundamental matrix, homography
#from autograd import numpy as np # gradient
import numpy as np
import math

import geometry

class OrientationEstimator:
    'Uses a view graph of keyframes to estimate current orientation.'
    def __init__(self, initial_image):
        self.initial_image = _Frame(initial_image)
    def add_image(self, img):
        'Returns updated estimate of current orientation.'
        current_image = _Frame(img)
        m1, m2 = _match(self.initial_image, current_image)
        rotation = _rotation(self.initial_image.keypoints[m1], current_image.keypoints[m2])
        return rotation # actually orientation, but initial_image defines origin

class _Frame:
    def __init__(self, img):
        self.keypoints, self.descriptors = _detect(img)

def _detect(img):
    'Returns keypoints and ORB descriptors of detected features.'
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, mask=None)
    # convert to something sane
    ret = np.empty((len(keypoints), 2), float)
    for i, kp in enumerate(keypoints):
        ret[i] = kp.pt
    return ret, descriptors

def _match(frame1, frame2):
    'Returns index array of matches.'
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(frame1.descriptors, frame2.descriptors)
    # convert to something sane
    ret1 = np.empty(len(matches), int)
    ret2 = np.empty(len(matches), int)
    for i, v in enumerate(matches):
        ret1[i] = v.queryIdx
        ret2[i] = v.trainIdx
    return ret1, ret2

def _rotation(p1, p2):
    'Returns estimated rotation from p1 to p2.'
    p1h = cv2.convertPointsToHomogeneous(p1).reshape((-1,3))
    p2h = cv2.convertPointsToHomogeneous(p2).reshape((-1,3))

    # estimate homography and fundamental matrix
    H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC)
    #print(H, np.count_nonzero(inliers)/len(p1))
    h_e = np.square(p2h - np.dot(H, p1h.T).T).sum(axis=1)
    gric_h = GRIC(h_e, 2, 8, 4)

    F, inliers = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)
    #print(F, np.count_nonzero(inliers)/len(p1))
    f_e = np.square(p2h - np.dot(F, p1h.T).T).sum(axis=1)
    gric_f = GRIC(f_e, 3, 7, 4)
    ret = "H", gric_h, "F", gric_f, (gric_f / gric_h) < 0.95

    return ret

def GRIC(e, d, k, r):
    # https://www.cs.ait.ac.th/~mdailey/papers/Tahir-KeyFrame.pdf
    lambda1 = math.log(r)
    lambda2 = math.log(r*len(e))
    lambda3 = 2.0
    sigma2 = 0.01
    ret = np.minimum( e/sigma2, lambda3 * (r - d) ).sum()
    ret += lambda1 * d * len(e)
    ret += lambda2 * k
    return ret

if __name__ == "__main__":
    cap = cv2.VideoCapture("pic/cam%03d.jpg")
    est = OrientationEstimator(cap.read()[1])
    i = 0
    while True:
        _, img = cap.read()
        if img is None:
            break
        r = est.add_image(img)
        print(i, *r)
        i += 1
