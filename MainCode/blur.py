from imutils import paths
import eyeinthesky as eye
import argparse
import cv2
import glob
import time
import sys
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
def Blur(image_path, show=False):
    check = True
    image0 = cv2.imread(image_path)
    image=eye.resize(15,image0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    # then the image should be considered "blurry
    if fm < 10:
        text = "Blurry"
        check = False
    # show the image
    if show:
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
    return fm

def batchAvg(folderPath, startTime=0, endTime=250):
    goodFiles={}
    goodFilesForPrint=[]
    frame=0
    validFrame=0
    tempBatch={}
    tempBatchForPrint={}
    lowerBound=startTime*30 #We are assuming 30 fps so 10 seconds would be 300 frames
    upperBound=endTime*30
    if sys.platform=='linux':
        maxBlur=700
    else:
        maxBlur=200
    allImgsLen=len(glob.glob(folderPath))
    for img in sorted(glob.glob(folderPath)):
        try:
            blurFac=Blur(img)
        except:
            frame+=1
            continue
        if blurFac<10 or blurFac>maxBlur or frame<lowerBound or frame>upperBound:
            frame+=1
            continue
        validFrame+=1
        tempBatch[img]=[frame,blurFac]
        tempBatchForPrint[frame]=blurFac
        if validFrame%10==0 or frame==upperBound:
            blurSum=0
            for k,value in tempBatch.items():
                blurSum+=value[1]
            avg=blurSum/len(tempBatch)
            for key, value in tempBatch.items():
                if value[1]<avg: continue
                else: goodFiles[value[0]]=key
            for key, value in tempBatchForPrint.items():
                if value<avg: continue
                else: goodFilesForPrint.append(key)
            print('{:.2f}:{}'.format(avg,tempBatch.values()))
            print('{}'.format(goodFilesForPrint))
            tempBatch={}
            goodFilesForPrint=[]
            tempBatchForPrint={}
        frame+=1
    print(goodFilesForPrint)
    return goodFiles


def test():
    startSim=5
    nums=[]
    for x in range(5):
        start=time.time()
        fNumber=startSim+x
        print(str(fNumber))
        if sys.platform=='linux':
            folder='/home/yehia/bagfiles/poster_recording/frames'+'/*.jpg'
        else:
            folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle'+str(fNumber)+r'\*.png'
        tot=0
        i=1
        num=[]
        for img in glob.glob(folder):
            i+=1
            start_time=time.time()
            num.append(Blur(img))
            end_time=time.time()-start_time
            print("Blur detection took {} second".format(end_time))
            tot+=num[-1]
        nums.append(num)
        print(f'TIME:{time.time()-start}; AVG:{tot/len(glob.glob(folder))}; HALF AVG:{(tot/len(glob.glob(folder)))*(3/4)}')
    print(f'{"FRAME":<5}|',end='')
    for x in range(len(nums)):
        print(f'{"SIM"+str(x+startSim):<8}|',end='')
    print('\n','-'*80)
    for x in range(250):
        print(f'{x+1:<5}|',end='')
        for y in range(len(nums)):
            print(f'{nums[y][x]:<8.3f}|',end='')
        print()
if __name__=="__main__":
    #test()
    if sys.platform=='linux':
        folder='/home/yehia/bagfiles/poster_recording/frames'+'/*.jpg'
    else:
        folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\Falling_wStyle8'+r'\*.png'
    validFrames=batchAvg(folder)
    print(validFrames)
    print(len(validFrames))