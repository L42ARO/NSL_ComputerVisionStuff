import glob
import sys,os
import eyeinthesky as eye
import numba
import math
import numpy as np
def main():
    focal_len=8.2
    if sys.platform=='linux':
        folder='rectdegtest/*.png'
        file2= 'newLS_sat_highQ.png'
    else:
        folder=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\90degtest\*.png'
        file2=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png'
        file3=r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\newLS_drone_5.png'
    i=1
    for img in glob.glob(folder):
        if(i%10==0):
            try:
                blockPrint()
                point=eye.getPoint(img,file2,showResults=False,whatToShow="All")
                enablePrint()
                print('FRAME {}: {}'.format(i,point), end=' ')
                check_ratio(point.img_kpts, point.sat_kpts, focal_len)
            except IndexError:
                    enablePrint()
                    print(f'FRAME {i}: No Matches',end=' ')
                    print(IndexError, end=' ')
            print('\n','-'*80)
        i+=1
# @numba.njit
def check_ratio(kpts1,kpts2,f):
    # print(kpts1)
    for x in range(len(kpts1)):
        if(x+2)>len(kpts1): break
        dist_px=math.dist(kpts1[x],kpts1[x+1])
        dist_w=math.dist(kpts2[x],kpts2[x+1])
        d=dist_w/dist_px
        print(d, end='; ')


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__=="__main__":
    main()