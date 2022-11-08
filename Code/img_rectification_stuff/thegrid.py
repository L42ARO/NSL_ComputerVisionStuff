import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import sys

def getGrid(file):
    img1 = cv.imread(file,1)          # queryImage
    dimensions = img1.shape
    # height, width, number of channels in image
    height = img1.shape[0]
    width = img1.shape[1]
    #print(height,width)
    color = (0, 255, 0)
    thickness = 2
    font = cv.FONT_HERSHEY_SIMPLEX
    grid_size=int(310/2)
    grid_count=0
    grid_list=[]
    for j in range(0,height,grid_size):
        for i in range(0,width,grid_size):
            start_point = (i, j)
            end_point = (i+grid_size, j+grid_size)
            distance=((((i+grid_size/2 - 1550 )**2) + ((j+grid_size/2 -1550)**2) )**0.5)
            if distance<1650:
                grid_count=grid_count+1
                grid_list.append([grid_count,i,i+grid_size,j,j+grid_size])
    #print(grid_list)
    return grid_list

def evalGrid(point,grid):
    for i in range(len(grid)):
        if point[0]>grid[i][1] and point[0]<grid[i][2]:
            if point[1]>grid[i][3] and point[1]<grid[i][4]:
                return int(grid[i][0])
        else:
            continue
    return -1

if __name__ == '__main__':
    if sys.platform=='linux':
        file='newLS_sat3_highQ.jpg'
    else:
        file=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat3_highQ.jpg'
    grid=getGrid(file)
    fGrid=evalGrid([157,2172],grid)
    print(fGrid)
