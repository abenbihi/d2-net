
import os
import numpy as np
import cv2


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

    
fe = cv2.ORB_create()
orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

NEW_SIZE = 1200, 300

SEQ = 3

if not os.path.exists('stat/%02d'%SEQ):
    os.makedirs('stat/%02d'%SEQ)

MACHINE = 1
if MACHINE == 1:
    DATA_ROOT_PATH = '/mnt/dataX/assia/kitti/dataset/sequences/%02d/image_2/'%SEQ
elif MACHINE == 2:
    DATA_ROOT_PATH = '/opt/BenbihiAssia/datasets/kitti/%02d/image_2/'%SEQ
else:
    print("stats_det: Once again, get your MTF MACHINE macro correct.")
    exit(1)

OUT_DIR = 'res/'
MIN_MATCH_COUNT = 10
EXT = '.d2-net'

img_fn_l = [l.split("\n")[0] for l in open("image_list.txt").readlines()]
img_num = len(img_fn_l)

for img_id, img_ref_fn in enumerate(img_fn_l):

    if img_id > img_num - 10:
        break


    # d2net detection
    out_ref_fn = '%s/%s%s'%(OUT_DIR, img_ref_fn, EXT)
    if not os.path.exists(out_ref_fn):
        continue
    data_ref = np.load(out_ref_fn)

    pts0 = data_ref['keypoints'] # kp_num, 3
    des0 = data_ref['descriptors'] # kp_num, des_dim
    sc0 = data_ref['scores']

    # convert to cv2 kp for prototype conformity
    kp0 = []
    for i, pt in enumerate(pts0):
        kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
            _angle=0, _response=0, _octave=0, _class_id=0)
        kp0.append(kp)

    # draw kp
    img0_fn = '%s/%s'%(DATA_ROOT_PATH,img_ref_fn)
    print(img0_fn)
    img0 = cv2.imread('%s/%s'%(DATA_ROOT_PATH,img_ref_fn))
    img0_bw = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    kp_on_img0 = np.tile(np.expand_dims(img0,2), (1,1,3))
    for i,kp in enumerate(kp0):
        pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(kp_on_img0, pt, 1, (0, 255, 0), -1, lineType=16)


    # orb
    orb_kp0, orb_des0 = fe.detectAndCompute(img0_bw,None)

    
    # write logs
    stat_fn = 'stat/%02d/%s.txt'%(SEQ, img_ref_fn.split(".")[0])
    stat_f = open(stat_fn, 'w')


    match_num = []
    for j in range(img_id+1,img_id+10):
        img_fn = img_fn_l[j]

        out_fn = '%s/%s%s'%(OUT_DIR, img_fn, EXT)
        data = np.load(out_fn)

        pts1 = data['keypoints']
        des1 = data['descriptors']
        sc1 = data['scores']

        # convert to cv2 kp for prototype conformity
        kp1 = []
        for i, pt in enumerate(pts1):
            kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
                _angle=0, _response=0, _octave=0, _class_id=0)
            kp1.append(kp)


        # draw kp
        img1 = cv2.imread('%s/%s'%(DATA_ROOT_PATH,img_fn))
        img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        kp_on_img1 = np.tile(np.expand_dims(img1,2), (1,1,3))
        for i,kp in enumerate(kp1):
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(kp_on_img1, pt, 1, (0, 255, 0), -1, lineType=16)


        # orb
        orb_kp1, orb_des1 = fe.detectAndCompute(img1_bw,None)


        # d2-net matches
        good = [] # matching features according to desc. distance
        matches = flann.knnMatch(des0, des1, k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
        match_des_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None, flags=2)
        
        if len(good)>MIN_MATCH_COUNT:
            kp1_good    = [ kp0[m.queryIdx] for m in good ]
            kp2_good    = [ kp1[m.trainIdx] for m in good ]
            kp1_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1_good])
            kp2_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp2_good])
            src_pts = np.float32(kp1_v).reshape(-1,1,2)
            dst_pts = np.float32(kp2_v).reshape(-1,1,2)
        
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,3.0)
            if F is None:
                print('No satisfying F was found :(')
                match_H_img_dict[img_key] = np.zeros((100,100)).astype(np.uint8)
                
            matchesMask = mask.ravel().tolist()
            draw_params = dict( matchesMask = matchesMask, flags = 2)
            match_H_img = cv2.drawMatches(img0, kp0, img1, kp1, good, None,**draw_params)
        else:
            print('sp len(good): %d'%(len(good)))
            match_H_img = np.zeros((NEW_SIZE[1], NEW_SIZE[0]*2)).astype(np.uint8)
        


        # orb matches
        orb_matches = orb_matcher.match(orb_des0, orb_des1)
        # Sort them in the order of their distance.
        # these are not necessarily good matches, I just called them
        # good to be homogeneous
        orb_good = sorted(orb_matches, key = lambda x:x.distance)
        orb_match_des_img = cv2.drawMatches(img0, orb_kp0, img1, orb_kp1, orb_good, None, 
                flags=2)
        
        if len(orb_good)>MIN_MATCH_COUNT:
            kp1_good    = [ orb_kp0[m.queryIdx] for m in orb_good ]
            kp2_good    = [ orb_kp1[m.trainIdx] for m in orb_good ]
            kp1_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1_good])
            kp2_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp2_good])
            src_pts = np.float32(kp1_v).reshape(-1,1,2)
            dst_pts = np.float32(kp2_v).reshape(-1,1,2)
        
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,3.0)
            if F is None:
                print('ORB: No satisfying F was found :(')
                orb_match_H_img_dict = np.zeros((100,100)).astype(np.uint8)
                
            orb_matchesMask = mask.ravel().tolist()
            draw_params = dict( matchesMask = orb_matchesMask, flags = 2)
            orb_match_H_img = cv2.drawMatches(img0, orb_kp0, img1, orb_kp1, orb_good, None,**draw_params)
        else:
            print('sp len(good): %d'%(len(good)))
            orb_match_H_img = np.zeros((NEW_SIZE[1], NEW_SIZE[0]*2)).astype(np.uint8)


        #cv2.imshow('match flann', match_des_img)
        #cv2.imshow('match F', match_H_img)
        #cv2.imshow('orb match flann', orb_match_des_img)
        #cv2.imshow('orb match F', orb_match_H_img)
        
        print('\nd2/orb: kp1: %d/%d\tkp2: %d/%d'%(len(kp0), len(orb_kp0),
            len(kp1), len(orb_kp1)))
        print('d2/orb: matches: %d/%d\tgood: %d/%d\transac: %d/%d'
                %(len(matches), len(orb_matches), len(good), len(orb_good), 
                    np.sum(matchesMask), np.sum(orb_matchesMask)))

        stat_f.write('%d, %d, %d, %d, '%(len(kp0), len(kp1), len(orb_kp0), len(orb_kp1)))
        stat_f.write('%d, %d, %d, %d, %d, %d\n'
                %(len(matches), len(good), np.sum(matchesMask), 
                    len(orb_matches),  len(orb_good), np.sum(orb_matchesMask)))

        #key = cv2.waitKey(0) & 0xFF
        #if key == ord('q'):
        #  print('Quitting, \'q\' pressed.')

    stat_f.close()
