import argparse
import os

import cv2
import numpy as np
import torch

#import tools

MIN_MATCH_COUNT = 50

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

MACHINE = 2
if MACHINE == 0:
    WS_DIR = "/home/gpu_user/assia/ws/"
elif MACHINE == 2:
    WS_DIR = "/opt/HomeLocauxTerm2/BenbihiAssia/ws/"
else:
    print("Get your MTF MACHINE macro ok.")
    exit(1)

PYDATA_DIR = "%s/datasets/pydata/pycmu/meta/"%WS_DIR

def draw_kp(img, pts):
    """
    Args:
        pts: 3xN shape (x,y,condifence)^T
    """
    if img.ndim == 2:
        kp_on_img = np.tile(np.expand_dims(img,2), (1,1,3))
    else:
        kp_on_img = img
    for i,pt in enumerate(pts):
        pt = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(kp_on_img, pt, 1, (0, 255, 0), -1, lineType=16)
    return kp_on_img


def draw_confidence(heatmap):
  min_conf = 0.001
  heatmap[heatmap < min_conf] = min_conf
  heatmap = -np.log(heatmap)
  heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
  out3 = tools.myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
  out32 = (out3*255).astype('uint8')
  return out32


def np2kp(pts):
    """Convert np array of detected points to list of cv::KeyPoint."""
    kp_l = []
    for pt in pts:
        kp = cv2.KeyPoint(x=pt[0],y=pt[1], _size=2,
            _angle=0, _response=0, _octave=0, _class_id=0)
        kp_l.append(kp)
    return kp_l


def resize_img(image):
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        s = args.max_edge / max(resized_image.shape) 
        resized_image = cv2.resize(image, None, fx=s, fy=s,
                interpolation=INTER_AREA)

        #resized_image = scipy.misc.imresize(
        #    resized_image,
        #    args.max_edge / max(resized_image.shape)
        #).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        s = args.max_sum_edges / sum(resized_image.shape[: 2])
        resized_image = cv2.resize(image, None, fx=s, fy=s,
                interpolation=INTER_AREA)

        #resized_image = scipy.misc.imresize(
        #    resized_image,
        #    args.max_sum_edges / sum(resized_image.shape[: 2])
        #).astype('float')
    resized_image = resized_image.astype(np.float32)
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    return resized_image, fact_i, fact_j


def resize_img_factor(image, factor):
    resized_image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return resized_image, 1/factor, 1/factor


def detectAndCompute(input_image, model):
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale( torch.tensor(
                input_image[np.newaxis, :, :, :].astype(np.float32),
                device=device), model)
        else:
            keypoints, scores, descriptors = process_multiscale( torch.tensor(
                input_image[np.newaxis, :, :, :].astype(np.float32),
            device=device), model, scales=[1])

    return keypoints, scores, descriptors


def match(args, img_pair_fn_l, K, d):
    """ """
    out_dir = "res/%s/"%args.fn_list.split("/")[-1].split(".")[0]
    
    # Creating CNN model
    model = D2Net(
        model_file=args.model_file,
        use_relu=args.use_relu,
        use_cuda=use_cuda
    )

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    for img_pair_fn in img_pair_fn_l:
        img0_fn, img1_fn = img_pair_fn.split(" ")
        print("img0_fn: %s\nimg1_fn: %s"%(img0_fn, img1_fn))
        img0 = cv2.imread(img0_fn)
        img1 = cv2.imread(img1_fn)
        if img0 is None:
            print("Error: failed to open %s"%img0_fn)
        if img1 is None:
            print("Error: failed to open %s"%img1_fn)


        # pre-proc for NN
        resized_img0, fact0_i, fact0_j = resize_img_factor(img0, 1.0)
        resized_img1, fact1_i, fact1_j = resize_img_factor(img1, 1.0)
        print(img0.shape)
        print(resized_img0.shape)
        print(fact0_i, fact0_j)
        #cv2.imshow("img0", img0)
        #cv2.imshow("resized_img0", resized_img0)
        #cv2.waitKey(0)

        img0_copy = img0.copy()
        img1_copy = img1.copy()

        resized_img0 = resized_img0[:,:,::-1] # bgr -> rgb (TODO: remove duplication)
        resized_img1 = resized_img1[:,:,::-1]

        input_img0 = preprocess_image(resized_img0, preprocessing=args.preprocessing)
        input_img1 = preprocess_image(resized_img1, preprocessing=args.preprocessing)

        pts0, scores0, des0 = detectAndCompute(input_img0, model)
        pts1, scores1, des1 = detectAndCompute(input_img1, model)

        # Input image coordinates
        pts0[:, 0] *= fact0_i
        pts0[:, 1] *= fact0_j
        # i, j -> u, v
        pts0 = pts0[:, [1, 0, 2]]

        pts1[:, 0] *= fact1_i
        pts1[:, 1] *= fact1_j
        # i, j -> u, v
        pts1 = pts1[:, [1, 0, 2]]

        kp0 = np2kp(pts0) # convert to cv keypoints
        kp1 = np2kp(pts1)
        print(pts0.shape)
        print(pts1.shape)
        #print(des0.shape)
        #print(des1.shape)
        #print("detection ok")

        if args.show_extra: 
            # draw kp on img
            kp_on_img0 = draw_kp(img0_copy, pts0) 
            kp_on_img1 = draw_kp(img1_copy, pts1) 
            #cv2.imshow("kp_on_img", np.hstack((kp_on_img0, kp_on_img1)))
            #cv2.waitKey(0)

            ## draw detection confidence
            #confidence0 = draw_confidence(heatmap0)
            #confidence1 = draw_confidence(heatmap1)
            #cv2.imshow("confidence", np.hstack((confidence0, confidence1)))
            #cv2.waitKey(0)
    
        ## matching features according to desc. distance
        #good = []
        #matches = flann.knnMatch(des0, des1, k=2)
        #for i,(m,n) in enumerate(matches):
        #    if m.distance < 0.95*n.distance:
        #        good.append(m)
        #    #good.append(m)
        #match_des_img = cv2.drawMatches(img0_copy, kp0, img1_copy, kp1, good, None, flags=2)
        #print("# of Lowe inliers: %d"%len(good))

        ## matches before fundamental estimation
        #out_des = []
        #for m in good:
        #    d = m.distance
        #    x0, y0 = kp0[m.queryIdx].pt
        #    x1, y1 = kp1[m.trainIdx].pt
        #    out_des.append([x0, y0, x1, y1, d])
        #out_des = np.vstack(out_des)


        ## select inliers with fundamental 
        #if len(good)>MIN_MATCH_COUNT:
        #    kp1_good    = [ kp0[m.queryIdx] for m in good ]
        #    kp2_good    = [ kp1[m.trainIdx] for m in good ]
        #    kp1_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1_good])
        #    kp2_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp2_good])
        #    src_pts = np.float32(kp1_v).reshape(-1,1,2)
        #    dst_pts = np.float32(kp2_v).reshape(-1,1,2)
        #
        #    #src_pts = cv2.undistortPoints(src_pts, K, d)
        #    #dst_pts = cv2.undistortPoints(dst_pts, K, d)
        #
        #    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,3.0)
        #    if F is None:
        #        print('No satisfying F was found :(')
        #        match_F_img_dict[img_key] = np.zeros((100,100)).astype(np.uint8)
        #    matchesMask = mask.ravel().tolist()
        #    print("# of F inliers: %d"%np.sum(np.array(matchesMask)))

        #    # matches after fundamental estimation
        #    out_F = []
        #    for i, m in enumerate(good):
        #        if matchesMask[i] == 1:
        #            #print(matchesMask[i])
        #            d = m.distance
        #            x0, y0 = kp0[m.queryIdx].pt
        #            x1, y1 = kp1[m.trainIdx].pt
        #            out_F.append([x0, y0, x1, y1, d])
        #    out_F = np.vstack(out_F)
        #    #print(out_F.shape)

        #    # draw new corners   
        #    draw_params = dict( matchesMask = matchesMask, flags = 2)
        #    match_F_img = cv2.drawMatches(img0_copy, kp0, img1_copy, kp1, good, None,**draw_params)
        #else:
        #    print('sp len(good): %d'%(len(good)))
        #    match_F_img = np.zeros((args.W, args.H*2)).astype(np.uint8)
        #
        #if (0==1): #args.show_extra: 
        #    cv2.imshow('match flann', match_des_img)
        #    cv2.imshow('match F', match_F_img)
        #    if (cv2.waitKey(0) & 0xFF) == ord("q"):
        #        exit(0)

        # save output for xiaolong
        kp_img_fn = "%s/img_kp/%s.jpg"%(out_dir, img1_fn.split("/")[-1].split(".")[0])
        cv2.imwrite(kp_img_fn, np.hstack((kp_on_img0, kp_on_img1)))
        
        #match_des_img_fn = "%s/img_match_raw/%s.jpg"%(out_dir, img1_fn.split("/")[-1].split(".")[0])
        #cv2.imwrite(match_des_img_fn, match_des_img)

        #match_F_img_fn = "%s/img_match_F/%s.jpg"%(out_dir, img1_fn.split("/")[-1].split(".")[0])
        #cv2.imwrite(match_F_img_fn, match_F_img)

        #match_des_fn = "%s/match_raw/%s.txt"%(out_dir, img1_fn.split("/")[-1].split(".")[0])
        #np.savetxt(match_des_fn, out_des, fmt="%.6f", header=" x0 y0 x1 y1 dist", comments="%")

        #match_F_fn = "%s/match_F/%s.txt"%(out_dir, img1_fn.split("/")[-1].split(".")[0])
        #np.savetxt(match_F_fn, out_F, fmt="%.6f", header=" x0 y0 x1 y1 dist",
        #        comments="%")


def match_efficient(args, model, db_fn, q_fn_l, out_dir):
    """ """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    print("\n\ndb_fn: %s"%db_fn)
    img0 = cv2.imread(db_fn)
    if img0 is None:
        print("Error: failed to open %s"%db_fn)
        exit(1)
    # pre-proc for NN
    resized_img0, fact0_i, fact0_j = resize_img_factor(img0, 1.0)
    img0_copy = img0.copy()
    resized_img0 = resized_img0[:,:,::-1] # bgr -> rgb (TODO: remove duplication)
    input_img0 = preprocess_image(resized_img0, preprocessing=args.preprocessing)
    
    # detection and description
    pts0, scores0, des0 = detectAndCompute(input_img0, model)
    # Input image coordinates
    pts0[:, 0] *= fact0_i
    pts0[:, 1] *= fact0_j
    # i, j -> u, v
    pts0 = pts0[:, [1, 0, 2]]

    kp0 = np2kp(pts0) # convert to cv keypoints
    if args.show_extra: 
        # draw kp on img
        kp_on_img0 = draw_kp(img0_copy, pts0)

    for q_fn in q_fn_l:
        print("\nq_fn: %s"%q_fn)
        img1 = cv2.imread(q_fn)
        if img1 is None:
            print("Error: failed to open %s"%q_fn)
            exit(1)

        # pre-proc for NN
        resized_img1, fact1_i, fact1_j = resize_img_factor(img1, 1.0)
        img1_copy = img1.copy()
        resized_img1 = resized_img1[:,:,::-1]
        input_img1 = preprocess_image(resized_img1, preprocessing=args.preprocessing)

        # detection and description
        pts1, scores1, des1 = detectAndCompute(input_img1, model)
        pts1[:, 0] *= fact1_i
        pts1[:, 1] *= fact1_j
        # i, j -> u, v
        pts1 = pts1[:, [1, 0, 2]]

        kp1 = np2kp(pts1)
        if args.show_extra: 
            # draw kp on img
            kp_on_img1 = draw_kp(img1_copy, pts1) 
            #cv2.imshow("kp_on_img", np.hstack((kp_on_img0, kp_on_img1)))
            #cv2.waitKey(0)

        # matching features according to desc. distance
        good = []
        matches = flann.knnMatch(des0, des1, k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.95*n.distance:
                good.append(m)
            #good.append(m)
        
        if len(good) == 0:
            match_des_img = np.zeros((100,100), np.uint8)
            match_F_img = np.zeros((100,100), np.uint8)
            out_des = np.zeros((1,2), np.int)
            out_F = np.zeros((1,2), np.int)
        else:
            match_des_img = cv2.drawMatches(img0_copy, kp0, img1_copy, kp1, good, None, flags=2)
            print("# of Lowe inliers: %d"%len(good))

            # matches before fundamental estimation
            out_des = []
            for m in good:
                d = m.distance
                x0, y0 = kp0[m.queryIdx].pt
                x1, y1 = kp1[m.trainIdx].pt
                out_des.append([x0, y0, x1, y1, d])
            out_des = np.vstack(out_des)

            
            if len(good)<=MIN_MATCH_COUNT:
                match_F_img = np.zeros((100,100), np.uint8)
                out_F = np.zeros((1,2), np.int)
                print('No satisfying F was found :(')
            else: # select inliers with fundamental
                kp1_good    = [ kp0[m.queryIdx] for m in good ]
                kp2_good    = [ kp1[m.trainIdx] for m in good ]
                kp1_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp1_good])
                kp2_v = np.vstack([[kp.pt[0],kp.pt[1]] for kp in kp2_good])
                src_pts = np.float32(kp1_v).reshape(-1,1,2)
                dst_pts = np.float32(kp2_v).reshape(-1,1,2)
            
                F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,3.0)
                if F is None:
                    match_F_img = np.zeros((100,100), np.uint8)
                    out_F = np.zeros((1,2), np.int)
                    print('No satisfying F was found :(')
                else:
                    matchesMask = mask.ravel().tolist()
                    print("# of F inliers: %d"%np.sum(np.array(matchesMask)))

                    # matches after fundamental estimation
                    out_F = []
                    for i, m in enumerate(good):
                        if matchesMask[i] == 1:
                            #print(matchesMask[i])
                            d = m.distance
                            x0, y0 = kp0[m.queryIdx].pt
                            x1, y1 = kp1[m.trainIdx].pt
                            out_F.append([x0, y0, x1, y1, d])
                    out_F = np.vstack(out_F)
                    #print(out_F.shape)

                    # draw new fundamental compliants matches
                    draw_params = dict( matchesMask = matchesMask, flags = 2)
                    match_F_img = cv2.drawMatches(img0_copy, kp0, img1_copy, kp1, good, None,**draw_params)
            
            if (0==1): #args.show_extra: 
                cv2.imshow('match flann', match_des_img)
                cv2.imshow('match F', match_F_img)
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    exit(0)

        # save output for xiaolong
        kp_img_fn = "%s/img_kp/%s.jpg"%(out_dir, q_fn.split("/")[-1].split(".")[0])
        cv2.imwrite(kp_img_fn, np.hstack((kp_on_img0, kp_on_img1)))
        
        match_des_img_fn = "%s/img_match_raw/%s.jpg"%(out_dir, q_fn.split("/")[-1].split(".")[0])
        cv2.imwrite(match_des_img_fn, match_des_img)

        match_F_img_fn = "%s/img_match_F/%s.jpg"%(out_dir, q_fn.split("/")[-1].split(".")[0])
        cv2.imwrite(match_F_img_fn, match_F_img)

        match_des_fn = "%s/match_raw/%s.txt"%(out_dir, q_fn.split("/")[-1].split(".")[0])
        np.savetxt(match_des_fn, out_des, fmt="%.6f", header=" x0 y0 x1 y1 dist", comments="%")

        match_F_fn = "%s/match_F/%s.txt"%(out_dir, q_fn.split("/")[-1].split(".")[0])
        np.savetxt(match_F_fn, out_F, fmt="%.6f", header=" x0 y0 x1 y1 dist",
                comments="%")


def update_list(args):
    # update names
    slice_id = 24
    fn_l = np.loadtxt(args.fn_list, dtype=str)
    new_fn_l = []
    for i, fn in enumerate(fn_l):
        q_fn = "slice%d/query/%s"%(slice_id, fn)
        db_fn = "slice%d/database/%s"%(slice_id, fn)
        if os.path.exists("%s/%s"%(args.img_dir, q_fn)):
            new_fn_l.append(q_fn)
            continue
        elif os.path.exists("%s/%s"%(args.img_dir, db_fn)):
            new_fn_l.append(db_fn)
            continue
        else:
            print("Error: this img does not belong to neither query nor database.")
            print("fn: %s\ndb_fn: %s/%s\nq_fn: %s/%s" %(fn, args.img_dir, db_fn, args.img_dir, q_fn))
            exit(1)

    np.savetxt("list0_new.txt", new_fn_l, fmt="%s")


def prepare_out_dir(args, list_fn):
    out_root_dir = "res/%d_c%d_db"%(args.slice_id, args.cam_id)
    for suffix in ["img_kp", "img_match_F", "img_match_raw", "match_F", "match_raw"]:
        out_dir = "%s/%s/%s/"%(out_root_dir, list_fn.split(".")[0], suffix)
        #print(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


def match_slice(args):
    """ """
    model = D2Net( model_file=args.model_file, use_relu=args.use_relu,
            use_cuda=use_cuda)

    list_dir = "%s/timelapses/%d_c%d_db/"%(PYDATA_DIR, args.slice_id, args.cam_id)
    for list_fn in sorted(os.listdir(list_dir)):
        print(list_fn)
        prepare_out_dir(args, list_fn)
        out_dir = "res/%d_c%d_db/%s/"%(args.slice_id, args.cam_id,
                list_fn.split(".")[0])

        fn_l = np.loadtxt("%s/%s"%(list_dir, list_fn), dtype=str)
        db_fn = "%s/%s"%(args.img_dir, fn_l[0])
        q_fn_l = ["%s/%s"%(args.img_dir, l) for l in fn_l[1:]]
        match_efficient(args, model, db_fn, q_fn_l, out_dir)


def main(args):
    fn_l = np.loadtxt(args.fn_list, dtype=str)
    db_fn = fn_l[0]
    img_pair_fn_l = []
    for fn in fn_l[1:]:
        img_pair_fn_l.append("%s/%s %s/%s"%(args.img_dir, db_fn, args.img_dir, fn))

    #for fn in img_pair_fn_l:
    #    print(fn)

    match(args, img_pair_fn_l, None, None) 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--slice_id', type=int, default=24)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--show_extra', action='store_true',
        help='Show extra debug outputs (default: False).')

    # Argument parsing
    parser.add_argument(
        '--fn_list', type=str, 
        help='path to a file containing a list of images to process'
    )
    
    parser.add_argument(
        '--preprocessing', type=str, default='caffe',
        help='image preprocessing (caffe or torch)'
    )
    parser.add_argument(
        '--model_file', type=str, default='models/d2_tf.pth',
        help='path to the full model'
    )
    
    parser.add_argument(
        '--max_edge', type=int, default=1600,
        help='maximum image size at network input'
    )
    parser.add_argument(
        '--max_sum_edges', type=int, default=2800,
        help='maximum sum of image sizes at network input'
    )
    
    parser.add_argument(
        '--output_extension', type=str, default='.d2-net',
        help='extension for the output'
    )
    parser.add_argument(
        '--output_type', type=str, default='npz',
        help='output file type (npz or mat)'
    )
    
    parser.add_argument(
        '--multiscale', dest='multiscale', action='store_true',
        help='extract multiscale features'
    )
    parser.set_defaults(multiscale=False)
    
    parser.add_argument(
        '--no-relu', dest='use_relu', action='store_false',
        help='remove ReLU after the dense feature extraction module'
    )
    parser.set_defaults(use_relu=True)
    args = parser.parse_args()
    
    print(args)

    #main(args)
    match_slice(args)
