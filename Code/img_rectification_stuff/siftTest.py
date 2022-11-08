import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import pywt
#approx 5.2 second running time

def resize(scale,img):
    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv.cvtColor( imArray,cv.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


start = time.time()
MIN_MATCH_COUNT = 0

img1 = cv.imread('/data/drone_pics/100.JPG',1)          # queryImage
img2 = cv.imread('/data/drone_pics/500.JPG',1)      # trainImage

# img1 = cv.imread('hamza_map1.jpg',1)
# img2 = cv.imread('hamza_map.jpg',1)      # trainImage

#
# img1 = cv.imread('alabama_map.png',0)
# img2 = cv.imread('alabama_uav.png',0)      # trainImage
#
# img1=cv.resize(img1,(2000, 2000), interpolation = cv.INTER_AREA)
# img2=cv.resize(img2,(2000, 2000), interpolation = cv.INTER_AREA)
#
# kernel = np.ones((5,5),np.uint8)

# img1 = w2d(img1,'haar',4)
# img2 = w2d(img2,'haar',4)


# img1=cv.GaussianBlur(img1,(5,5),0)
# img2=cv.GaussianBlur(img2,(5,5),0)

# img1 = cv.medianBlur(img1,5)
# img2 = cv.medianBlur(img2,5)
# img1 = cv.adaptiveThreshold(img1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
# img2 = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

# img1= cv.Canny(img1,50,75)
# img2= cv.Canny(img2,70,75)

# ret, img1 = cv.threshold(img1,127,255,cv.THRESH_BINARY)
# ret, img2 = cv.threshold(img2,127,255,cv.THRESH_BINARY)

# img1 = cv.GaussianBlur(img1,(5,5),0)
# img2 = cv.GaussianBlur(img2,(5,5),0)
# ret3,img1 = cv.threshold(img1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# ret3,img2 = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#
# img1 = cv.GaussianBlur(img1,(5,5),0)
# img2 = cv.GaussianBlur(img2,(5,5),0)
# img1 = cv.adaptiveThreshold(img1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
# img2 = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)



# img1= cv.morphologyEx(img1, cv.MORPH_GRADIENT, kernel)
# img2= cv.morphologyEx(img2, cv.MORPH_GRADIENT, kernel)


# Initiate SIFT detector
sift = cv.SIFT_create()
fast = cv.FastFeatureDetector_create()
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# kp1 = fast.detect(img1,None)
# kp2 = fast.detect(img2,None)

# kp1 = orb.detect(img1,None)
# kp1, des1 = orb.compute(img1, kp1)
#
# kp2 = orb.detect(img2,None)
# kp2, des2 = orb.compute(img2, kp2)


#BFMatcher with default params
bf = cv.BFMatcher()


matches = bf.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

# Apply ratio test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

# cv.drawMatchesKnn expects list of lists as matches.




draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 2)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
img4 = cv.drawKeypoints(img1, kp1, None, color=(255,0,0))
img5 = cv.drawKeypoints(img2, kp2, None, color=(255,0,0))
plt.figure()
plt.imshow(img3)
plt.figure()
plt.imshow(img4)
plt.figure()
plt.imshow(img5)
plt.show()

end = time.time()
print(f"Runtime of the program is {end - start}")