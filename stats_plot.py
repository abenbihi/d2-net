
import os
import numpy as np
import cv2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SEQ = 3

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

    # logs
    stat_fn = 'stat/%02d/%s.txt'%(SEQ, img_ref_fn.split(".")[0])
    fig_fn = 'stat/%02d/%s'%(SEQ, img_ref_fn)

    stat = np.loadtxt(stat_fn, delimiter=',')
    kp0_v, kp1_v = stat[:,0], stat[:,1]
    orb_kp0_v, orb_kp1_v = stat[:,2], stat[:,3]
    matches, good, matchesMask = stat[:,4], stat[:,5], stat[:,6]
    orb_matches, orb_good, orb_matchesMask = stat[:,7], stat[:,8], stat[:,9]


    absc = np.arange(kp0_v.shape[0])

    plt.figure(1, figsize=(10, 3))
    G = gridspec.GridSpec(1, 1)
    
    ax1 = plt.subplot(G[0, 0])
    ax1.set_ylabel('Match number')
    ax1.set_xlabel('Step')
    ax1.set_title('Matches: d2net vs orb')
    ax1.plot(absc, matchesMask, color='r', label='ransac - d2net', alpha=0.3)
    ax1.plot(absc, orb_matchesMask, color='b', label='ransac - orb', alpha=0.3)
    
    ax1.plot(absc, good, 'ro', label='lowe - d2net', alpha=0.3)
    ax1.plot(absc, orb_good, 'bo', label='lowe - orb', alpha=0.3)
    
    ax1.plot(absc, matches, 'r*', label='d2net', alpha=0.3)
    ax1.plot(absc, orb_matches, 'b*', label='orb', alpha=0.3)

    ax1.legend(loc='upper right')


    plt.savefig(fig_fn)
    plt.close()
    toto = cv2.imread(fig_fn)
    cv2.imshow('fig', toto)
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        exit(0)





