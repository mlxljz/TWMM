import cv2
import numpy as np
import matlab.engine

import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *




eng = matlab.engine.start_matlab()


def trans2DArray2matlab(array_2D,matlab_type='uint8'):
    list_2D =array_2D.tolist()
    assert  matlab_type=='uint8' or matlab_type=='uint16' or matlab_type=='double',matlab_type
    if matlab_type=='uint8':
        mat_array=matlab.uint8(list_2D)
    elif matlab_type=='uint16':
        mat_array = matlab.uint16(list_2D)
    elif matlab_type == 'double':
        mat_array = matlab.double(list_2D)
    else:
        return None
    return mat_array

def findHomoraphy(good):
    try:
        MIN_MATCH_COUNT = 4
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([[_[0], _[1]] for _ in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([[_[2], _[3]] for _ in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            M = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float64)
        if M is None:
            print("M is None - {}/{}".format(len(good), MIN_MATCH_COUNT))
            M = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float64)

        flag=True
        if len(good)<20 or abs(M[0][2])>60 or abs(M[1][2])>60 or abs(M[0][0])>1.2 or abs(M[0][0])<0.8 or abs(M[0][1])>1.2 or abs(M[1][0])>1.2 or abs(M[1][1])>1.2 or abs(M[1][1])<0.8 or abs(M[2][0])>0.05 or abs(M[2][1])>0.05:
            print('good_match is:',len(good))
            flag=False

    except:
        flag = False
        good=[]
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float64)

    return M,flag,len(good)

def sift_registration(img1, img2):
    img1gray = cv2.normalize(img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    img2gray = img2

    sift = cv2.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    MIN_MATCH_COUNT = 4
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float64)
    if M is None:
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float64)
    return M,(pts1,pts2)


def hardnet_registration(img1,img2):
    def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor):
        if len(cv2_sift_kpts) == 0:
            return np.array([])

        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            kornia_descriptor.eval()
            timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float() / 255.)
            lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts)
            patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
            B, N, CH, H, W = patches.size()
            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
        return descs.detach().cpu().numpy()


    def sift_korniadesc_matching(fname1, fname2, descriptor):
        # img1 = cv2.normalize(fname1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        # img2 = fname2
        img1 = fname1#cv2.cvtColor(fname1, cv2.COLOR_BGR2RGB)
        img2 = fname2#cv2.cvtColor(fname2, cv2.COLOR_BGR2RGB)

        sift = cv2.xfeatures2d_SIFT().create(8000)#8000的意思是最多保留8000个关键点，少于8000则都保留，多于8000则只保留8000个
        kps1 = sift.detect(img1, None)
        kps2 = sift.detect(img2, None)
        # That is the only change in the pipeline -- descriptors
        descs1 = get_local_descriptors(img1, kps1, descriptor)
        descs2 = get_local_descriptors(img2, kps2, descriptor)
        # The rest is the same, as for SIFT

        dists, idxs = KF.match_smnn(torch.from_numpy(descs1), torch.from_numpy(descs2), 0.75)
        tentatives = cv2_matches_from_kornia(dists, idxs)
        src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1,1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1,1, 2)

        pts1=[]
        pts2=[]
        for m in tentatives:
            pts1.append(kps1[m.queryIdx].pt)
            pts2.append(kps2[m.trainIdx].pt)

        if len(tentatives)>4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            M=None
        # F, inliers_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 0.75, 0.99, 100000)

        # draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
        #                    singlePointColor=None,
        #                    matchesMask=inliers_mask.ravel().tolist(),  # draw only inliers
        #                    flags=2)
        # img_out = cv2.drawMatches(img1, kps1, img2, kps2, tentatives, None, **draw_params)
        return M,(pts1,pts2)

    hardnet = KF.HardNet(True)
    return sift_korniadesc_matching(img1,img2,hardnet)

def tfeat_registration(img1,img2):
    def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor):
        if len(cv2_sift_kpts) == 0:
            return np.array([])

        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            kornia_descriptor.eval()
            timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float() / 255.)
            lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts)
            patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
            B, N, CH, H, W = patches.size()
            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
        return descs.detach().cpu().numpy()


    def sift_korniadesc_matching(fname1, fname2, descriptor):
        # img1 = cv2.normalize(fname1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        # img2 = fname2
        img1 = fname1#cv2.cvtColor(fname1, cv2.COLOR_BGR2RGB)
        img2 = fname2#cv2.cvtColor(fname2, cv2.COLOR_BGR2RGB)

        sift = cv2.xfeatures2d_SIFT().create(8000)
        kps1 = sift.detect(img1, None)
        kps2 = sift.detect(img2, None)
        # That is the only change in the pipeline -- descriptors
        descs1 = get_local_descriptors(img1, kps1, descriptor)
        descs2 = get_local_descriptors(img2, kps2, descriptor)
        # The rest is the same, as for SIFT

        dists, idxs = KF.match_smnn(torch.from_numpy(descs1), torch.from_numpy(descs2), 0.75)#0.9是first to second nearest neighbor distance的阈值
        tentatives = cv2_matches_from_kornia(dists, idxs)
        src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1,1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1,1, 2)

        pts1=[]
        pts2=[]
        for m in tentatives:
            pts1.append(kps1[m.queryIdx].pt)
            pts2.append(kps2[m.trainIdx].pt)

        if len(tentatives)>4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            M=None
        # F, inliers_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 0.75, 0.99, 100000)

        # draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
        #                    singlePointColor=None,
        #                    matchesMask=inliers_mask.ravel().tolist(),  # draw only inliers
        #                    flags=2)
        # img_out = cv2.drawMatches(img1, kps1, img2, kps2, tentatives, None, **draw_params)
        return M,(pts1,pts2)

    tfeat = KF.TFeat(True)
    return sift_korniadesc_matching(img1,img2,tfeat)






def orb_registration(img1, img2):
    img1gray =cv2.normalize(img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    img2gray = img2
    orb = cv2.ORB_create()
    kpts1, descs1 = orb.detectAndCompute(img1gray, None)
    kpts2, descs2  = orb.detectAndCompute(img2gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    pts1 = []
    pts2 = []
    for m in dmatches:
        pts1.append(kpts1[m.queryIdx].pt)
        pts2.append(kpts2[m.trainIdx].pt)

    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M,(pts1, pts2)

def surf_registration(img1,img2):
    img1gray = cv2.normalize(img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    img2gray = img2

    surf = cv2.xfeatures2d_SURF().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1gray, None)
    kp2, des2 = surf.detectAndCompute(img2gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75* n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    MIN_MATCH_COUNT = 4
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float64)
    return M, (pts1,pts2)

def scb_registration(img1,img2):
    eng.cd('feature_based/SCB_registration')
    H= eng.SCB_demo_python(trans2DArray2matlab(img1, matlab_type='uint16'),trans2DArray2matlab(img2, matlab_type='uint16'))
    return  H,None



def rift_registration(img1,img2):
    eng.cd('feature_based/RIFT_registration')
    good=eng.RIFT_demo_python(trans2DArray2matlab(img1, matlab_type='uint16'),
                        trans2DArray2matlab(img2, matlab_type='uint8'))
    good=[[_[0],_[1],_[2],_[3]] for _ in good]
    H_mat, flag_result, good_match = findHomoraphy(good)
    src_pts = np.float32([[_[0], _[1]] for _ in good]).reshape(-1,2)
    dst_pts = np.float32([[_[2], _[3]] for _ in good]).reshape(-1, 2)
    return H_mat,(src_pts,dst_pts)
