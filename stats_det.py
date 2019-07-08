
import numpy as np

DATA_ROOT_PATH = '/opt/BenbihiAssia/datasets/kitti/01/image_2/'
OUT_DIR = 'res/'


img_fn_l = [l.split("\n")[0] for l in open("image_list.txt").readlines()]
img_num = len(img_fn_l)

for i, img_ref_fn in enumerate(img_fn_l):

    if i > img_num - 10:
        break

    out_ref_fn = '%s/%s/%s'%(OUT_DIR, img_ref_fn, EXT)
    data_ref = np.load(out_ref_fn)

    kp0 = data_ref['keypoints']
    des0 = data_ref['descriptors']
    sc0 = data_ref['scores']

    for j in range(i,i+10):
        img_fn = img_fn_l[j]

        out_fn = '%s/%s/%s'%(OUT_DIR, img_fn, EXT)
        data = np.load(out_fn)

        kp1 = data['keypoints']
        des1 = data['descriptors']
        sc1 = data['scores']



    

    path = line.strip()

    out_fn = OUT_DIR + path + args.output_extension
    data = np.load(out_fn)
    
    kp = a['keypoints']
    des = a['descriptors']
    score = a['scores']

