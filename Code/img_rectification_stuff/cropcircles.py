from tkinter import W
import eyeinthesky as eye
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import glob
import sys, os
def main():
    #eye.getPoint(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\newLS_drone_4.png',r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png', False, "All")
    point=[]
    i=1
    quadCounts={'1':0,'2':0,'3':0,'4':0}
    confList=[]
    rateFac=10
    for img in glob.glob(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle2\*.png'):
        if(i>=250): break
        elif(i>=200):
            rateFac=2
        if(i%rateFac==0):
            try:
                blockPrint()
                point.append(eye.getPoint(img,r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png', False, "All"))
                enablePrint()
                print(f'FRAME: {i} --> {point[-1]}', end=' ')
                quadCounts[str(point[-1].quadrant)]+=1
                point[-1].percentFall=(i/250)**2
                confList.append(point[-1].confidence)
                print(f'--> SUCCESS SO FAR: {(quadCounts["4"]/len(point)*100):.2f}%\n','-'*120)
            except IndexError:
                print(f'FRAME: {i} No Matches',end=' ')
            except:
                print(f'FRAME: {i} Failed because WTF?', end=' ')
        i+=1
    quadMode=int(max(quadCounts, key=quadCounts.get))
    confMax=max(confList)
    totalMap=cv.imread(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png')
    totalWidth=totalMap.shape[1]
    totalHeight=totalMap.shape[0]
    fimg=plt
    sumAxis={'x':0,'y':0, 'total':0}
    print(f'SUCCESS RATE: {quadCounts["4"]/len(point)*100:.2f}%')
    for c in point:
        wFac=totalWidth/2 if (c.quadrant==2 or c.quadrant==4) else 0
        hFac=totalHeight/2 if (c.quadrant>2) else 0
        nonRel_coords=[c.sat_coords[0]/0.35+wFac, c.sat_coords[1]/0.35+hFac]
        fimg.plot(nonRel_coords[0], nonRel_coords[1], 'ro')
        if(c.quadrant==quadMode):
            sumAxis['x']+=nonRel_coords[0]*(c.confidence/confMax)*(c.percentFall)
            sumAxis['y']+=nonRel_coords[1]*(c.confidence/confMax)*(c.percentFall)
            sumAxis['total']+=1*(c.confidence/confMax)*(c.percentFall)
    fimg.plot(sumAxis['x']/sumAxis['total'], sumAxis['y']/sumAxis['total'], 'bo')
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