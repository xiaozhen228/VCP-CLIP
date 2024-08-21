import numpy as np 
import os 
import cv2
from sklearn.metrics import auc
from skimage import measure

def cal_iou(gt,pre):
    ground_truth = gt.astype(np.uint8)
    prediction = pre.astype(np.uint8)
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def normalize(pred, max_value=None, min_value=None):

    if max_value is None or min_value is None:
        if (pred.max() - pred.min()) == 0:
            return np.zeros_like(pred)
        else:
            return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def he_cheng(img_list, size = 256):
    h,w,c = img_list[0].shape
    jian = np.ones((h, 10, 3),dtype=np.uint8) * 255
    vis_con = img_list[0]
    for i in range(1,len(img_list)):
        vis_con = np.concatenate([vis_con, jian, img_list[i]], axis=1)

    vis_con = cv2.resize(vis_con, (size*len(img_list)+ 10*(len(img_list)-1), size)).astype(np.uint8)
    return vis_con


def visualization(save_root, pic_name, raw_image, raw_anomaly_map, raw_gt, the = 0.5, size = 518):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    pic_name = pic_name.replace("bmp", "png")

    
    assert len(raw_image.shape) == 3 and len(raw_anomaly_map.shape) == 2 and len(raw_gt.shape) == 2
    map = raw_anomaly_map
    gt = raw_gt

    #np.save(os.path.join(save_root, "text_"+pic_name.replace('bmp', 'npy')), text)
    #np.save(os.path.join(save_root, "vis_map_"+pic_name.replace('bmp', 'npy')), map)
    #np.save(os.path.join(save_root, "gt_"+pic_name.replace('bmp', 'npy')), gt)

    
    
    img = cv2.cvtColor(raw_image , cv2.COLOR_BGR2RGB)
    map = normalize(raw_anomaly_map)
    gt = normalize(raw_gt)
    map_binary = np.array(raw_anomaly_map> the, dtype= np.uint8)
    map_crop = map * map_binary

    ground_truth_contours, _ = cv2.findContours(np.array(raw_gt * 255, dtype = np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis_map = apply_ad_scoremap(img, map)
    vis_gt = apply_ad_scoremap(img, gt)
    vis_map_binary = apply_ad_scoremap(img, map_binary)
    vis_map_crop = apply_ad_scoremap(img, map_crop)

    vis_map = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR)
    vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
    vis_map_binary = cv2.cvtColor(vis_map_binary, cv2.COLOR_RGB2BGR)
    vis_map_crop = cv2.cvtColor(vis_map_crop, cv2.COLOR_RGB2BGR)

    vis_map_binary = cv2.drawContours(vis_map_binary, ground_truth_contours, -1, (0, 255, 0), 2)
    vis_map_crop = cv2.drawContours(vis_map_crop, ground_truth_contours, -1, (0, 255, 0), 2)  

    zong = he_cheng([raw_image, vis_map, vis_map_crop, vis_gt])
    #cv2.imwrite(os.path.join(save_root, "vis_map_"+pic_name), vis_map)
    #cv2.imwrite(os.path.join(save_root, "vis_gt_"+pic_name), vis_gt)
    #cv2.imwrite(os.path.join(save_root, "vis_map_binary_"+pic_name), vis_map_binary)
    #cv2.imwrite(os.path.join(save_root, "vis_map_crop_"+pic_name), vis_map_crop)
    cv2.imwrite(os.path.join(save_root, "vis_zong_"+pic_name), zong)
    