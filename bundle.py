from autograd import numpy as np
from math import tan, radians
from autograd import value_and_grad

# todo: what is the best way to encode orientation?

def reprojection_error(points, camera, views, observations):
    assert len(views) == len(observations)
    ret = 0.0
    camera_matrix = cam2mat(camera)
    for view, (pointids, observed_pixels) in zip(views, observations):
        position, orientation = view[:3], view[3:]
        camera_centric = rotate(points[pointids], orientation) - position
        expected_pixels = project(camera_centric, camera_matrix)
        ret = ret + np.linalg.norm(observed_pixels - expected_pixels, axis=1)
    return ret

def rotate(points, q):
    rot = quat2mat(q)
    return np.dot(points, rot)

_FLOAT_EPS = np.finfo(np.float).eps
def quat2mat(q):
    # ------- https://github.com/matthew-brett/transforms3d BSD -----------
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def cam2mat(camera):
    # http://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#details
    fovh, width, height = camera
    f = width/2 * tan(radians(fovh)/2)
    cx, cy = width/2, height/2
    return np.array(
        ((f, 0, cx),
         (0, f, cy),
         (0, 0,  1)))

def project(points, camera_matrix):
    projected = np.dot(points, camera_matrix.T)
    divided = projected / projected[:,2].reshape((len(projected),1))
    assert (divided[:,2] == 1).all()
    return divided[:,0:2]

def test_project():
    w, h = 1280, 962
    camera_matrix = cam2mat((90, w, h))
    res = np.array([
        [ w/2,    0,  w/2 ],
        [   0,  w/2,  h/2 ],
        [   0,    0,    1 ]])
    assert np.allclose(camera_matrix, res)
    test_set = np.array([
        [ 0, 0,1,    w/2, h/2],
        [ 1, 0,1,      w, h/2],
        [-1, 0,1,      0, h/2],
        [ 0, h/w,1,  w/2,   h],
        [ 0,-h/w,1,  w/2,   0],
    ], dtype=float)
    points = test_set[:,0:3]
    pixels = project(points, camera_matrix)
    assert np.allclose(pixels, test_set[:,3:5])

def test_rotate():
    points = np.ones((1,3))
    e = np.array([1,0,0,0]) # identity quat
    eye = quat2mat(e)
    assert np.allclose(np.identity(3), eye)
    assert (rotate(points, e) == points).all()
    quat = [0, 1, 0, 0]
    mat = quat2mat(quat) # 180 degree rotation around x axis
    assert np.allclose(mat, np.diag([1, -1, -1]))
    rotated = rotate(points, quat) # flips y and z
    assert np.allclose(rotated, [[ 1, -1, -1]])

def test_reprojection_error():
    points = np.zeros((1,3))
    camera = 60, 640, 480 # fovh, w, h
    views = np.array([[0.,0.,-1., 1.,0.,0.,0.]]) # origin, no rotation
    observations = [ [[0], np.array([[640/2., 480/2.]])] ]
    error = reprojection_error(points, camera, views, observations)
    assert np.allclose(error, 0)

    points = np.zeros((1,3))
    points[0][2] += 0.1 # move by 10 cm further from camera, this should not increase error
    error = reprojection_error(points, camera, views, observations)
    assert np.allclose(error, 0)

    points = np.zeros((1,3))
    points[0][0] += 0.1 # move by 10 cm to the right (x axis), increases error
    error = reprojection_error(points, camera, views, observations)
    assert error > 0.001

    gfun = value_and_grad(reprojection_error)
    error_wrapped, gradient = gfun(points, camera, views, observations)
    assert error == error_wrapped # function is properly wrapped

def test_gradient_descent():
    points = np.zeros((1,3))
    camera = 60, 640, 480 # fovh, w, h
    views = np.array([[0.,0.,-1., 1.,0.,0.,0.]]) # origin, no rotation
    observations = [ [[0], np.array([[640/2., 480/2.]])] ]

    points[0][0] += 0.1

    gfun = value_and_grad(reprojection_error)
    error_before, gradient = gfun(points, camera, views, observations)
    points -= gradient/10000
    error_after, gradient = gfun(points, camera, views, observations)
    assert error_after < error_before # error decreses
    assert points[0][0] < 0.1 # point is moved towards zero

if __name__ == "__main__":
    test_project()
    test_rotate()
    test_reprojection_error()
    test_gradient_descent()
