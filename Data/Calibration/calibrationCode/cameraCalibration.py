import numpy as np
import cv2 as cv
import glob
chessboardSize= (24,17)
frameSize=(1440, 1080)

criteria=(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER,30,0.001)

objp= np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2]=np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objpoints=[]
imgpoints=[]

images= glob.glob(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\calibrationCode\*.png')
for image in images:
    print(image)
    img=cv.imread(image)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners=cv.findChessboardCorners(gray,chessboardSize,None)
    if ret==True:
        objpoints.append(objp)
        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img,chessboardSize,corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(1000)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs=cv.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
print('CameraCalibrated: ',ret)
print('CameraMatrix: \n',mtx)
print('DistCoeffs: \n',dist)
print('RrotationVectors: \n',rvecs)
print('TranslationVectors: \n',tvecs)

np.savez(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\CameraParams.npz',mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
