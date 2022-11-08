import cv2
import numpy as np


file3=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\CameraParams.npz'
with np.load(file3) as file:
    mtx0,dist0,rvecs0,tvecs0 = [file[i] for i in ('mtx','dist','rvecs','tvecs')]
points_2D = np.array([
                        (950,810),  #Green Cube
                        (1320,560),  # Yellow cube
                        (595, 560),  #Blue Cube
                        (960, 390),  # Red Cube
 
                      ], dtype="double")
points_3D = np.array([
 
                      (0.5,0.5,0),  #Green Cube
                      (-0.5,0.5,0),  # Yellow cube
                      (0.5, -0.5,0),  #Blue Cube
                      (-0.5,-0.5,0),  # Red Cube
                     ])
ret, rvec,tvec=cv2.solvePnP(points_3D,points_2D,mtx0,dist0)
rotM=cv2.Rodrigues(rvec)[0]
cameraPos=-np.matrix(rotM).T*np.matrix(tvec)
print(cameraPos)

