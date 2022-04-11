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
from numba import jit

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
        return f'{self.img_coords[0]:>5.2f},{self.img_coords[1]:>5.2f} --> {self.sat_coords[0]:>5.2f},{self.sat_coords[1]:>5.2f} --> {self.quadrant} --> {self.confidence:>5.2f}'
# FUNCTION TO LOCALLY TEST THE MODULE ---------------------------------------------------------
def __test():
    coords=getPoint(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\3D_sim_tests\newLS_drone_4.png',r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\NewLSTemplates\newLS_sat_highQ.png', True, "All")
    print(coords)
# FUNCTION THAT GETS THE MIDPOINT AND KEYPOINTS OF THE IMAGE ---------------------------------------------------------
def getPoint(rocketImage, satImage, showResults=False, whatToShow="All"):
    with np.load(r'C:\Users\L42ARO\Documents\USF\SOAR\NSL_ComputerVisionStuff\Data\Calibration\CameraParams.npz') as f:
        mtx0,dst0,rvecs0,tvecs0 = [f[i] for i in ('mtx','dist','rvecs','tvecs')]
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
        nImg,oImg=load_torch_image(loadedImg,a*90)
        vidImg.append(nImg)
        ogImg.append(oImg)
    mapImg=subdivisions(fname2,35)
    p_acr, p_mkpts, p_inliers=quadIter(vidImg, mapImg)
    f=p_acr.index(max(p_acr))    
    f_img1 = vidImg[f-int(f/4)*4]
    f_img2 = mapImg[int(f/4)]
    f_inliers=np.array(p_inliers[f])
    f_mkpts0=np.array(p_mkpts[f][0])
    f_mkpts1=np.array(p_mkpts[f][1])

    print(f'The final countdown: {time.time()-start}')

    print(f'Winner: {f} --> Quadrant {int(f/4)+1}')
    print(len(f_mkpts1))
    print(len(f_inliers))
    print(type(int(f/4)+1))
    f_keypoints ,accImg_kpts, accSat_kpts= mid_points(f_mkpts0,f_mkpts1,f_inliers)
    final=f_refpoints(f_keypoints[0],f_keypoints[1], accImg_kpts,accSat_kpts, int(f/4)+1,max(p_acr),ogImg[f-int(f/4)*4]) #WE CREATE AN OBJECT THAT HOLDS THE DATA TO RETURN
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
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    acc=[0,0,0,0]
    bett_acc=[0,0,0,0]
    true_inliers=[]
    true_mkpoints=[]
    for t in range(4):
        print(f'{t*90} deg:',end=' ')
        tstImg = rotImgs[t]
        quadImg = img2
        matcher = KF.LoFTR(pretrained='outdoor')
        #MATCHING STARTS-------------------------------------------------------------------------------
        matcher = matcher.eval().cuda()
        #if(device=='cuda:0'):
        input_dict = {"image0": K.color.rgb_to_grayscale(tstImg), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(quadImg)}
        with torch.no_grad():
            correspondences = matcher(input_dict)
        #FILTER OUT RESULTS-----------------------------------------------------------------------------
        filtParams=[torch.count_nonzero(correspondences['confidence']>0.5),torch.mean(correspondences['confidence']),list(correspondences['confidence'].size())]
        print(f"TotPts: {filtParams[2]}; PtsWConf>0.5: {filtParams[0]}; avgConf: {filtParams[1]:.3f};", end=' ')        
        if(filtParams[1]<0.37 or filtParams[2][0]<8):
            true_inliers.append([])
            true_mkpoints.append([])
            acc[t]=0
            print()
            continue
        #GETTING TRUE INLIERS--------------------------------------------------------------------------
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        true_mkpoints.append([mkpts0, mkpts1])
        H1, inliers1 = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.9999, 100000)
        inliers1 = inliers1 > 0
        H2, inliers2 = pydegensac.findFundamentalMatrix(mkpts0, mkpts1, 0.4)
        print(f'USAC match: {np.count_nonzero(inliers1 == True)};', end=' ')
        print(f'DEGENSAC match: {np.count_nonzero(inliers2 == True)};', end=' ')
        inliers_i= []
        for i,k in enumerate(inliers1): #CROSS CHECKING BETWEEN USAC AND DEGENSAC
            if(k[0]==True and inliers2[i]==True):
                inliers_i.append(True)
                acc[t]+=1
            else:
                inliers_i.append(False)
        print(f'Acc match: {acc[t]};') #PRINTING RAW ACCURACY
        true_inliers.append(inliers_i)
        bett_acc[t]=float(filtParams[1]*acc[t])
        if acc[t]>20:
            break     
    return bett_acc,true_mkpoints, true_inliers
 # FUNCTION THAT ITERATES THROUGH THE MAP QUADRANTS AND EVALUATES ---------------------------------------------------------
def quadIter(img1List=[], img2List=[]):
    prt_acr=[]
    prt_mkpts=[]
    prt_inliers=[]
    for q in range(4):
        print(f'QUADRANT {q+1}:')
        start2=time.time()
        n_acr,n_mkpts,n_inliers=iterRot(img1List,img2List[q])
        prt_acr.extend(n_acr)
        prt_mkpts+=n_mkpts
        prt_inliers+=n_inliers
        print(f'Time after time for Q{q+1}: {time.time()-start2}')
    return prt_acr, prt_mkpts, prt_inliers
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
            accImg_kpts.append(n)
            total2 = total2 + n[1]
    for v in range(len(y)):
        if z[v]:
            v = y[v]
            accSat_kpts.append(v)
            total3 = total3 + v[0]
    for j in range(len(y)):
        if z[j]:
            m = y[j]
            accSat_kpts.append(m)
            total4 = total4 + m[1]
            filter1+=1
    xyt = [(1/filter1)*total, (1/filter1)*total2]
    xyt2 = [(1/filter1)*total3, (1/filter1)*total4]
    return [xyt, xyt2], accImg_kpts, accSat_kpts


if __name__=="__main__":
    __test()
