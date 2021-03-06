import matplotlib.pyplot as plt
import cv2
import kornia.feature as KF
import kornia as K
import numpy as np
import torch
from kornia_moons.feature import *
import time
import imutils
import pydegensac
from numba import njit
from numba import jit
import sys, os

device=''
# OBJECT THAT HOLDS THE DATA TO RETURN ---------------------------------------------------------
class f_refpoints:
    def __init__(self, imgCoords,satCoords,imgkpts, satkpts, Q, conf, og):
        self.img_coords=imgCoords
        self.sat_coords=satCoords
        self.img_kpts=imgkpts
        self.sat_kpts=satkpts
        self.quadrant=Q
        self.confidence=conf
        self.percentFall=1
        self.og_img=og
    def __str__(self):
        return f'{self.img_coords[0]:>5.2f},{self.img_coords[1]:>5.2f} --> {self.sat_coords[0]:>5.2f},{self.sat_coords[1]:>5.2f} --> {self.quadrant} --> {self.confidence:>5.3f}'
# FUNCTION TO LOCALLY TEST THE MODULE ---------------------------------------------------------
def __test():
    if sys.platform=='linux':
        file1='newLS_drone_4.png'
        file2= 'newLS_sat_highQ.png'
    else:
        base=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file1=base+'/Data/3D_sim_tests/Falling_wStyle6/newLS_drone_anim7_0188.png'
        file2=base+'/Data/NewLSTemplates/newLS_sat_4-4_highQ.png'
    coords=getPoint(file1,file2, [1,2,3,4], True, "All", True)
    print(coords)
# FUNCTION THAT GETS THE MIDPOINT AND KEYPOINTS OF THE IMAGE ---------------------------------------------------------
def getPoint(rocketImage, satImage, Qorder=[1,2,3,4], showResults=False, whatToShow="All", changeParams=False):
    # blockPrint()
    global device
    if(device==''):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start=time.time()
    fname1 = rocketImage
    fname2= satImage
    vidImg=[]
    ogImg=[]
    loadedImg=resize(15,cv2.imread(fname1))
    for a in range(4):
        nImg,oImg=load_torch_image(loadedImg,a*85)
        vidImg.append(nImg)
        ogImg.append(oImg)
    mapImg=subdivisions(fname2,25)
    p_acr, p_mkpts=iterQuad(vidImg, mapImg,Qorder)
    max_acr=[max(p_acr)]
    p_acr2=p_acr.copy()
    k_mkpts0=[]
    k_mkpts1=[]
    k_inliers=[]
    mat=[]
    for x in range(2 if changeParams else 1):
        f=p_acr.index(max_acr[x])
        f_q=Qorder[int(f/4)]
        f_r=f-int(f/4)*4
        print(f'Conf:{p_acr[f]} --> Quad:{f_q}')
        k_mkpts0.append(np.array(p_mkpts[str(f_q)][f_r][0]))
        k_mkpts1.append(np.array(p_mkpts[str(f_q)][f_r][1]))
        a_inliers,amat=cleanMatches(k_mkpts0[x],k_mkpts1[x])
        k_inliers.append(a_inliers)
        mat.append(amat)
        p_acr2.remove(max_acr[x])
        max_acr.append(max(p_acr2))
        if not(amat<3 or (abs(max_acr[x]-max_acr[x+1])<2)):break
    # enablePrint()
    idx=mat.index(max(mat))
    if(max(mat)<2):
        raise IndexError
    f=p_acr.index(max_acr[idx])
    f_q=Qorder[int(f/4)]
    f_r=f-int(f/4)*4
    f_img1 = vidImg[f_r]
    f_img2 = mapImg[f_q-1]
    f_mkpts0=k_mkpts0[idx]
    f_mkpts1=k_mkpts1[idx]
    f_inliers=k_inliers[idx]
    print(f'Winner: {f} --> Quadrant {f_q}')
    f_keypoints ,accImg_kpts, accSat_kpts= mid_points(f_mkpts0,f_mkpts1,f_inliers)
    final=f_refpoints(f_keypoints[0],f_keypoints[1], accImg_kpts,accSat_kpts, f_q,p_acr[f],ogImg[f_r]) #WE CREATE AN OBJECT THAT HOLDS THE DATA TO RETURN
    print(f'The final countdown: {time.time()-start}')
    if (whatToShow=="Midpoint"):    
        f_mkpts0 = np.array([f_keypoints[0]])
        f_mkpts1 =  np.array([f_keypoints[1]])
        f_inliers =  np.array([True])
    if(showResults):
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(f_mkpts0).view(1,-1, 2),
                                        torch.ones(f_mkpts0.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(f_mkpts0.shape[0]).view(1,-1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(f_mkpts1).view(1,-1, 2),
                                        torch.ones(f_mkpts1.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(f_mkpts1.shape[0]).view(1,-1, 1)),
            torch.arange(f_mkpts0.shape[0]).view(-1,1).repeat(1,2),
            K.tensor_to_image(f_img1),
            K.tensor_to_image(f_img2),
            f_inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None, 
                       'feature_color': (0.2, 0.5, 1), 'vertical': False})
        plt.show()
    return final

def cleanMatches(mkpts0,mkpts1):
    matches=0
    H1, inliers1 = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.45, 0.9999, 100000)
    inliers1 = inliers1 > 0
    H2, inliers2 = pydegensac.findFundamentalMatrix(mkpts0, mkpts1, 0.45)
    H3, inliers3 = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_DEFAULT, 0.45, 0.9999, 100000)
    inliers3 = inliers3 > 0
    print(f'USAC match: {np.count_nonzero(inliers1 == True)};', end=' ')
    print(f'DEGENSAC match: {np.count_nonzero(inliers2 == True)};', end=' ')
    print(f'USAC_ACCURATE match: {np.count_nonzero(inliers3 == True)}', end=' ')
    inliers_i,matches=crossCheck(inliers1,inliers2,inliers3)
    print(f'Acc match: {matches};') #PRINTING RAW ACCURACY
    return inliers_i, matches
@njit
def crossCheck(i1,i2,i3):
    inliers_i=[]
    matches=0
    for i in range(len(i1)): #CROSS CHECKING BETWEEN USAC AND DEGENSAC
        if(i1[i][0]==True and i2[i]==True and i3[i]==True):
            inliers_i.append(True)
            matches+=1
        else:
            inliers_i.append(False)
    return np.array(inliers_i), matches

def undistortImg(img,mtx,dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
# FUNCTION THAT RESIZES THE IMAGAES ---------------------------------------------------------
def resize(scale,img):
    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized
# FUNCTION THAT CONVERTS IMAGESS TO TORCH IMAGES ---------------------------------------------------------
def load_torch_image(timg, rot):
    global device
    if (device==""):
        device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img= timg
    if rot!=0:
        img = imutils.rotate_bound(img, rot)
    ogimg=img.copy()
    print("img read & resized")
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    img=img.to(device)
    print("loaded img")
    return img,ogimg
# FUNCTION THAT ROTATES THE IMAGE AND EVALUATES ---------------------------------------------------------\

def iterRot(rotImgs=[],img2=None):
    global device
    acc=[0,0,0,0]
    true_mkpoints=[]
    for t in range(4):
        tstImg = rotImgs[t]
        quadImg = img2
        matcher = KF.LoFTR(pretrained='outdoor')
        #MATCHING STARTS-------------------------------------------------------------------------------
        if(str(device)=='cuda:0'):
            matcher = matcher.eval().cuda()
        input_dict = {"image0": K.color.rgb_to_grayscale(tstImg), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(quadImg)}
        with torch.no_grad():
            correspondences = matcherFunc(matcher, input_dict)
        #FILTER OUT RESULTS-----------------------------------------------------------------------------
        filtParams=[torch.count_nonzero(correspondences['confidence']>0.6),torch.mean(correspondences['confidence']),list(correspondences['confidence'].size())]
        print('{} deg:'.format(t*90),end=' ')
        print("TotPts: {}; PtsWConf>0.5: {}; avgConf: {:.3f};".format(filtParams[2],filtParams[0],filtParams[1]), end=' ')        
        if(filtParams[1]<0.34 or filtParams[2][0]<8):
            true_mkpoints.append([])
            acc[t]=0
            print()
            continue
        #GETTING TRUE INLIERS--------------------------------------------------------------------------
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        true_mkpoints.append([mkpts0, mkpts1])   
        acc[t]=float(filtParams[1]*filtParams[2][0])
        print('Acc: {:.3f}'.format(acc[t]))
        if(acc[t]>15):
            break
    return acc,true_mkpoints
# @torch.jit.script
def matcherFunc(mat, inDict):
    return mat(inDict)
 # FUNCTION THAT ITERATES THROUGH THE MAP QUADRANTS AND EVALUATES ---------------------------------------------------------
def iterQuad(img1List=[], img2List=[], order=[1,2,3,4]):
    prt_acr=[]
    prt_mkpts={"1":[],"2":[],"3":[],"4":[]}
    for q in order:
        print(f'QUADRANT {q}:')
        start2=time.time()
        n_acr,n_mkpts=iterRot(img1List,img2List[q-1])
        prt_acr.extend(n_acr)
        prt_mkpts[str(q)]=n_mkpts
        print(f'Time after time for Q{q}: {time.time()-start2}')
    return prt_acr, prt_mkpts

# FUNCTION THAT SUBDIVIDES THE MAP INTO THE QUADRANTS ---------------------------------------------------------
def subdivisions(img2Dir, scale):
    global device
    if (device==""):
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = cv2.imread(img2Dir)
    (h, w) = image.shape[:2]
    # compute the center coordinate of the image
    (cX, cY) = (w // 2, h // 2)
    #cv2.imshow('Original', image)
    imgs=[image[0:cY, 0:cX],image[0:cY, cX:w],image[cY:h, 0:cX],image[cY:h, cX:w]]
    new_imgs=[]
    for i,k in enumerate(imgs):
        new_imgs.append(resize(scale,k))
        new_imgs[i]=K.image_to_tensor(new_imgs[i], False).float() /255.
        new_imgs[i]=K.color.bgr_to_rgb(new_imgs[i])
        new_imgs[i]=new_imgs[i].to(device)
    return new_imgs
# FUNCTION THAT FINDS THE MIDPOINT GIVEN KEPOINTS ---------------------------------------------------------
@njit
def mid_points(x,y,z):
    total = 0
    total2 = 0
    total3 = 0
    total4 = 0
    filter1 = 0
    accImg_kpts=[]
    accSat_kpts=[]
    for i in range(len(x)):
        if z[i]:
            s = x[i]
            accImg_kpts.append(s)
            total = total + s[0]
    for k in range(len(x)):
        if z[k]:
            n = x[k]
            total2 = total2 + n[1]
    for v in range(len(y)):
        if z[v]:
            u = y[v]
            accSat_kpts.append(u)
            total3 = total3 + u[0]
    for j in range(len(y)):
        if z[j]:
            m = y[j]
            total4 = total4 + m[1]
            filter1+=1
    xyt = [(1/filter1)*total, (1/filter1)*total2]
    xyt2 = [(1/filter1)*total3, (1/filter1)*total4]
    return [xyt, xyt2], accImg_kpts, accSat_kpts

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__=="__main__":
    __test()
