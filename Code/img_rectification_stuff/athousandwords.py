from cgitb import enable
from tkinter import W

from cv2 import SOLVEPNP_DLS, SOLVEPNP_IPPE, SOLVEPNP_ITERATIVE, findHomography
import eyeinthesky as eye
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import glob
import numpy as np
import sys, os
import cupy as cp
def main():
    for n in range(5):
        with np.load(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\CameraParams.npz') as file:
            mtx0,dst0,rvecs0,tvecs0 = [file[i] for i in ('mtx','dist','rvecs','tvecs')]
        realWorldWidth=2532.35
        realWorldHeight=2096.82
        blockPrint()
        point=eye.getPoint(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle2\newLS_drone_anim2_0039.png',r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png', False, "All")
        enablePrint()
        print('-'*120)
        print(f'POINT-->{point}')
        print('-'*120)
        totalMap=cv.imread(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png')
        testImg=cv.imread(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle2\newLS_drone_anim2_0039.png')
        totalWidth=totalMap.shape[1]
        totalHeight=totalMap.shape[0]
        avgRealWorldRatio=(realWorldWidth/totalWidth+realWorldHeight/totalHeight)/2
        fimg=plt
        fimg.figure(n)
        realWorld_kpts=[]
        nonRelSat_kpts=[]
        initialpt=[totalWidth*avgRealWorldRatio,totalHeight*avgRealWorldRatio]
        for x in point.sat_kpts:
            wFac=totalWidth/2 if (point.quadrant==2 or point.quadrant==4) else 0
            hFac=totalHeight/2 if (point.quadrant>2) else 0
            nonRelSat_kpts.append([x[0]/0.35+wFac,x[1]/0.35+hFac,0])
            fimg.plot(nonRelSat_kpts[-1][0], nonRelSat_kpts[-1][1], 'ro')
            realWorld_kpts.append([nonRelSat_kpts[-1][0]*avgRealWorldRatio,nonRelSat_kpts[-1][1]*avgRealWorldRatio,0])
            if(nonRelSat_kpts[-1][0]<initialpt[0]):
                initialpt[0]=nonRelSat_kpts[-1][0]
            if(nonRelSat_kpts[-1][1]<initialpt[1]):
                initialpt[1]=nonRelSat_kpts[-1][1]
        fimg.plot(initialpt[0], initialpt[1], 'go')
        for i,s in enumerate(nonRelSat_kpts):
            nonRelSat_kpts[i][0]-=initialpt[0]
            nonRelSat_kpts[i][1]-=initialpt[1]
            fimg.plot(nonRelSat_kpts[i][0], nonRelSat_kpts[i][1], 'bo')
        realWorld_kpts=np.array(realWorld_kpts,dtype=np.float32)
        point.img_kpts=np.array(point.img_kpts,dtype=np.float32)
        nonRelSat_kpts=np.array(nonRelSat_kpts,dtype=np.float32)
        realPointsList=[nonRelSat_kpts]
        imgPointsList=[point.img_kpts]
        blockPrint()
        print(f'RealWorld_kpts: {len(realWorld_kpts)}, Img_kpts: {len(point.img_kpts)}')
        for x in range(3):
            if x==0:
                ret, mtx, dist, rvec, tvec=cv.calibrateCamera(realPointsList, imgPointsList, point.og_img.shape[:2], None, None)
                print(f'camera matrix: {mtx}')
                rotM=cv.Rodrigues(rvec[0])[0]
                cameraPos=-np.matrix(rotM).T*np.matrix(tvec[0])
            if x==1:
                ret,rvec, tvec=cv.solvePnP(nonRelSat_kpts, point.img_kpts, mtx0, dst0, flags=SOLVEPNP_ITERATIVE)
                rotM=cv.Rodrigues(rvec)[0]
                cameraPos=-np.matrix(rotM).T*np.matrix(tvec)
            fimg.plot(cameraPos[0,0], cameraPos[1,0],'co' if x==1 else 'yo')
            print(f'rvecs: {rvec}')
            print(f'tvecs: {tvec}')
        enablePrint()
        h,  w = point.og_img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        H,state=cv.findHomography(point.img_kpts, nonRelSat_kpts)
        H = H.T
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]
        K_inv = np.linalg.inv(newcameramtx)
        L = 1 / np.linalg.norm(np.dot(K_inv, h1))
        r1 = L * np.dot(K_inv, h1)
        r2 = L * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        T = L * (K_inv @ h3.reshape(3, 1))
        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))
        print(f'Possible sln: {T}')
        fimg.imshow(totalMap)
    fimg.show()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__=="__main__":
    main()

#with special thanks to R the man J