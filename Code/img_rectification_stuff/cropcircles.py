from cgitb import enable
from tkinter import W
import eyeinthesky as eye
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import glob
import sys, os
import time
from numba import njit
def main():
    #eye.getPoint(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\newLS_drone_4.png',r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png', False, "All")
    if sys.platform=='linux':
        folder='Falling_wStyle2/*.png'
        file2= 'newLS_sat_highQ.png'
    else:
        folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle6\*.png'
        file2=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat2_highQ.png'
    startTime=time.time()
    totalMap=cv.imread(file2)
    totalWidth=totalMap.shape[1]
    totalHeight=totalMap.shape[0]
    point=[]
    i=1
    quadCounts={'1':0,'2':0,'3':0,'4':0}
    confList=[]
    rateFac=1
    nextOrder=[1,2,3,4]
    for img in glob.glob(folder):
        if(i>200): break
        if(i%rateFac==0):
            try:
                blockPrint()
                point.append(eye.getPoint(img,file2, nextOrder, showResults=False, whatToShow="All", changeParams=False if i<160 else True))
                enablePrint()
                print(f'FRAME: {i} --> {point[-1]}', end=' ')
                quadCounts[str(point[-1].quadrant)]+=1
                point[-1].percentFall=i
                print(f'--> {point[-1].percentFall:.2f}', end=' ')
                confList.append(point[-1].confidence)
                nextOrder=sortNext(point[-1].sat_coords, point[-1].quadrant,totalWidth,totalHeight)
                print(f'--> {nextOrder}', end=' ')
                print(f'--> SUCCESS SO FAR: {(quadCounts["4"]/len(point)*100):.2f}%')
            except IndexError:
                enablePrint()
                print(f'FRAME: {i} No Matches')
            print('-'*120)
        i+=1
    quadMode=int(max(quadCounts, key=quadCounts.get))
    # confMax=max(confList)
    
    fimg=plt
    sumAxis={'x':0,'y':0, 'total':0}
    print(f'SUCCESS RATE: {quadCounts["4"]/len(point)*100:.2f}%')
    for c in point:
        wFac=totalWidth/2 if (c.quadrant==2 or c.quadrant==4) else 0
        hFac=totalHeight/2 if (c.quadrant>2) else 0
        nonRel_coords=[c.sat_coords[0]/0.35+wFac, c.sat_coords[1]/0.35+hFac]
        fimg.plot(nonRel_coords[0], nonRel_coords[1], 'ro')
        fimg.text(nonRel_coords[0], nonRel_coords[1], str(c.percentFall), fontsize=10)
        if(c.quadrant==quadMode):
            sumAxis['x']+=nonRel_coords[0]
            sumAxis['y']+=nonRel_coords[1]
            sumAxis['total']+=1
    print(f'TOTAL TIME: {time.time()-startTime:.2f}')        
    fimg.plot(sumAxis['x']/sumAxis['total'], sumAxis['y']/sumAxis['total'], 'bo')
    fimg.imshow(totalMap)
    fimg.show()

@njit
def sortNext(sat_coords,quadrant,totW,totH):
    imgMidW=(totW/4)*0.35 #It's over 4 because that the midpoint
    imgMidH=(totH/4)*0.35
    order=[quadrant]
    if(sat_coords[0]<imgMidW):
        if(sat_coords[1]<imgMidH):
            subQ=1
        else:
            subQ=2
    else:
        if(sat_coords[1]<imgMidH):
            subQ=3
        else:
            subQ=4
    if(quadrant==1):
        if(subQ<3):
            order.extend([2,3,4])
        else:
            order.extend([3,2,4])
    elif(quadrant==2):
        if(subQ<3):
            order.extend([1,4,3])
        else:
            order.extend([4,1,3])
    elif(quadrant==3):
        if(subQ<3):
            order.extend([1,4,2])
        else:
            order.extend([4,1,2])
    elif(quadrant==4):
        if(subQ<3):
            order.extend([2,3,1])
        else:
            order.extend([3,2,1])
    return order
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    

if __name__=="__main__":
    main()
        
#with special thanks to R the man J