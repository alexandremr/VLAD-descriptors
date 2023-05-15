import cv2


# doc at http://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#gsc.tab=0
def describeSURF(image):
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(400)
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors


# doc at http://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#gsc.tab=0
def describeSIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def describeORB(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
