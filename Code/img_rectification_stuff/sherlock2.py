
import eyeinthesky as eye
import blur as blr
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import sys, os
import time
from numba import njit
import warnings
def main():
    desiredQuadrant=4
    desiredSubQuadrant=1
    MainTimer=Clock() #We Start the timer to test the function runTime
    Test=TestValues(rateFac=100, L2_Th=800)#note: since we are anlayzing 1330 imgs it should be around 665 plus the first set of images that are worthless
    Test.defineFiles('Falling_wStyle8','newLS_sat2_highQ.png') #Files must be relative to the Data folder
  #  TotalMap=AreaMap(Test.sat_file) #We get the total map that wil be used to plot the results
    allFrames=[]
    processed=0
    for img in sorted(glob.glob(Test.folder)):

        if(processed>3000): break
        if not(Test.proceedEval(img)):
            #print('wtf: ', Test.blur_param)
            continue
        try:
            blockPrint()
            point=eye.getPoint(img,'newLS_sat2_highQ.png', Test.nextOrder, showResults=False, whatToShow="Midpoint", changeParams=False)
            Frame=FrameTest(point,Test.currFrame, Test.blur_param)
            enablePrint()
            Test.nextOrder, Frame.subQ=sortNext(Frame.sat_coords, Frame.quadrant,Frame.og_map.shape[1],Frame.og_map.shape[0])
            Test.quadCounts[str(Frame.quadrant)]+=1
            Test.subQuadCounts[str(Frame.quadrant)][str(Frame.subQ)]+=1
            Test.confList.append(Frame.confidence)
            allFrames.append(Frame)
            print(Frame, end=' ')
            print(f'--> {Test.nextOrder}--> Q_SUCCESS: {(Test.quadCounts[str(desiredQuadrant)]/len(allFrames)*100):.2f}% --> subQ_SUCCESS: {(Test.subQuadCounts[str(desiredQuadrant)][str(desiredSubQuadrant)]/len(allFrames)*100):.2f}%')
            processed+=1
        except IndexError:
            enablePrint()
            print(f'FRAME: {Test.currFrame} | BLUR: {Test.blur_param:>5.3f} | x x x x NO MATCHES x x x x')
            processed+=1
        print('-'*120)
        Test.currFrame+=1
        if(Test.currFrame>=Test.L2_Threshold) and not Test.L2_ModeSet:
            Test.L2_ModeSet=True
            print('*'*120)
            Test.currMode[0]=int(max(Test.quadCounts, key=Test.quadCounts.get))
            Test.currMode[1]=int(max(Test.subQuadCounts[str(Test.currMode[0])], key=Test.subQuadCounts[str(Test.currMode[0])].get))
            # Test.defineFiles('Falling_wStyle8','Level02/newLS_sat2_highQ')
            print(f'MODE: {Test.currMode}')
            print('*'*120)
    print(MainTimer, "; processed frames: ", processed)
    f=open("timeResult1.txt","a")
    t=str(time.time()-MainTimer.startTime)+","+str(processed)
    f.write(t)
    f.close()

class Clock:
    def __init__(self):
        self.startTime=time.time()
    def getTime(self):
        return time.time()-self.startTime
    def __str__(self):
        return f'TIME: {time.time()-self.startTime}'
class AreaMap:
    def __init__(self, file2):
        self.map=cv.imread(file2)
        self.width=self.map.shape[1]*0.25
        self.height=self.map.shape[0]*0.25
class TestValues:
    def __init__(self, rateFac=10, L2_Th=120):
        self.quadCounts={'1':0,'2':0,'3':0,'4':0}
        self.subQuadCounts={'1':{'1':0,'2':0,'3':0,'4':0},'2':{'1':0,'2':0,'3':0,'4':0},'3':{'1':0,'2':0,'3':0,'4':0},'4':{'1':0,'2':0,'3':0,'4':0}}
        self.confList=[]
        self.currMode=[0,0]
        self.rateFac=rateFac
        self.currFrame=0
        self.nextOrder=[1,2,3,4]
        self.evalNext=False
        self.L2_Threshold=L2_Th
        self.L2_ModeSet=False
        self.goodFrames=[]
        #Other variables that are defined later
        #self.folder,self.sat_file, self.blur_param, self.subQCounts
    def defineFiles(self,fold,f2):
        if sys.platform=='linux':
            print("We are using linux, good luck loser")
            self.folder='/home/yehia/bagfiles/poster_recording/frames'+'/*.jpg'
            self.sat_file= '/home/yehia/opencvstuff/newLS_sat2_highQ.png'#
        else:
            base=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.folder=base+'/Data/3D_sim_tests/'+fold+'/*.png'
            self.sat_file=base+'/Data/NewLSTemplates/'+f2+'.png'
    def proceedEval(self,img):
        if(self.currFrame%self.rateFac==0 or self.evalNext):
            if self.isImgGood(img):
                self.evalNext=False
                return True
            else:
                self.evalNext=True
                self.currFrame+=1
                return False
        else:
            self.currFrame+=1
            return False

    def isImgGood(self,img):
        self.blur_param=blr.Blur(img)
        if(img in self.goodFrames):
            return True
        else:
           return False
class FrameTest(eye.f_refpoints):
    def __init__ (self, point, currFrame, blur_param):
        super().__init__(point.img_coords, point.sat_coords, point.img_kpts, point.sat_kpts, point.quadrant, point.confidence, point.og_img, point.og_map)
        self.frameNum=currFrame
        self.frameBlur=blur_param
        #Other variables that are defined later:
        self.subQ=1

    def __str__(self):
        return f'FRAME: {self.frameNum} | BLUR: {self.frameBlur:>5.3f} | QUADRANT: {self.quadrant} | SUBQUADRANT: {self.subQ} | CONFIDENCE: {self.confidence:>5.2f}'
# return f'{self.img_coords[0]:>5.2f},{self.img_coords[1]:>5.2f} --> {self.sat_coords[0]:>5.2f},{self.sat_coords[1]:>5.2f} --> {self.quadrant} --> {self.confidence:>5.3f}'
@njit
def sortNext(sat_coords,quadrant,totW,totH):
    imgMidW=(totW/2) #It's over 4 because that the midpoint
    imgMidH=(totH/2)
    order=[quadrant]
    if(sat_coords[0]<imgMidW):
        if(sat_coords[1]<imgMidH):
            subQ=1
        else:
            subQ=3
    else:
        if(sat_coords[1]<imgMidH):
            subQ=2
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
    return order, subQ
def blockPrint():
    warnings.filterwarnings("ignore")
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    #warnings.simplefilter('module')
    sys.stdout = sys.__stdout__


if __name__=="__main__":
    main()