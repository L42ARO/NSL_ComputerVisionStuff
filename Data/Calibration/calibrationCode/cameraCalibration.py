import numpy as np
import cv2 as cv
import glob
import sys
chessboardSize= (14,11)#chessboardSize= (24,17)
frameSize=(1440, 1080)
if sys.platform== 'linux':
    folder='/home/yehia/Launch_4152022/Frames/*.jpg'
    saveFileName='/home/yehia/opencvstuff/CameraParams2.npz'
else:
    #folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\calibrationCode\*.png'
    folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\calibrationCode\Real_Camera\Calibration\*.jpg'
    saveFileName=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\CameraParams2.npz'


criteria=(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER,30,0.001)

objp= np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2]=np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objpoints=[]
imgpoints=[]
images= glob.glob(folder)
for image in images:
    print(image)
    img=cv.imread(image)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners=cv.findChessboardCorners(gray,chessboardSize,None)
    print(ret)
    if ret==True:
        objpoints.append(objp)
        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        try:
            cv.drawChessboardCorners(img,chessboardSize,corners2,ret)
            cv.imshow('img',img)
            cv.waitKey(1000)
        except:
            print("Couldn't display IMGS")

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs=cv.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
print('CameraCalibrated: ',ret)
print('CameraMatrix: \n',mtx)
print('DistCoeffs: \n',dist)
print('RrotationVectors: \n',rvecs)
print('TranslationVectors: \n',tvecs)

f=8.2
sx=3.6
sy=2.7
w=870
h=720
print(f'Check accuracy:\
    \n[{w*f/sx:.1f} 0 {w/2:.1f}]\
    \n[0 {h*f/sy:.1f} {h/2:.1f}]\
    \n[0 0 1]')

np.savez(saveFileName,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
