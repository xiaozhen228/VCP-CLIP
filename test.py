import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from utils.dataset import VisaDataset, MVTecDataset, OtherDataset
from models.prompt_ensemble import Prompt_Ensemble
from tqdm import tqdm
from models.pre_vcp import Context_Prompting
from models.post_vcp import Zero_Parameter
from models.model_CLIP import Load_CLIP, tokenize
from tqdm import tqdm
from torch import nn
from scipy.ndimage import gaussian_filter
from utils.tools import cal_iou, normalize, visualization, cal_pro_score, auc

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        assert 'ViT' in model
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
    def forward(self, tokens):
        tokens_list = []
        for i in range(len(tokens)):
            tokens_list.append(self.fc[i](tokens[i][:, 1:, :]))
        return tokens_list
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_stages(model, params, exclude_key=None):
    for n, m in model.named_parameters():
        if exclude_key and exclude_key in n:
            assert m.data.size() == params[n].data.size()
            m.data = params[n].data


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    model , preprocess_train , preprocess_test = Load_CLIP(img_size, args.pretrained_path , device=device, deep_prompt_len = args.deep_prompt_len, total_d_layer_len = args.total_d_layer_len)
    model.to(device)
    model.eval()
    
    tokenizer = tokenize

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)


    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)
    linearlayer.eval()
    
    New_Lan_Embed = Context_Prompting(model_configs, cla_len= args.prompt_len).to(device)
    New_Lan_Embed.eval()
    prompt_ceshi = Prompt_Ensemble( args.prompt_len,tokenizer)


    Zero_try = Zero_Parameter(dim_v = model_configs["vision_cfg"]['width'], dim_t =  model_configs['text_cfg']['width'], dim_out= model_configs["vision_cfg"]['width']).to(device)
    Zero_try.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location= device)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"], strict=False)
    New_Lan_Embed.load_state_dict(checkpoint["New_Lan_Embed"])
    Zero_try.load_state_dict(checkpoint["Zero_try"], strict=False)
    _load_stages(model, checkpoint, "prompt")

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])


    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess_test, target_transform=transform,
                                aug_rate=-1, mode='test',  train_mode="zero",dataset=dataset_name)
    elif dataset_name == 'visa':
        test_data = VisaDataset(root=dataset_dir, transform=preprocess_test, target_transform=transform, mode='test', train_mode="zero",dataset=dataset_name)
    else:
        test_data = OtherDataset(root=dataset_dir, transform=preprocess_test, target_transform=transform, mode='test', train_mode="zero", dataset=dataset_name)
        
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    obj_list = test_data.get_cls_names()


    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['pr_sp'] = []
    results['gt_sp'] = []
    results['anomaly_map_raw'] = []
    results['anomaly_map_new'] = []
    results['path'] = []
    id = 0
    test_dataloader = tqdm(test_dataloader)
    for items in test_dataloader:
        id = id + 1
        #print(id)
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask.squeeze(1).numpy())  # px
        results['gt_sp'].extend(items['anomaly'])
        

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, features_list)

            class_token = New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings = prompt_ceshi.forward_ensemble(model, class_token, device)

            text_embeddings = text_embeddings.permute(0,2,1)
            anomaly_maps_new = []

            for layer in range(len(patch_tokens)):
                dense_feature = patch_tokens[layer][:,1:,:].clone()
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                F_s_a, F_t_a = Zero_try(text_embeddings.permute(0,2,1), dense_feature)
                anomaly_map_new = (Zero_try.prompt_temp_l1.exp() * dense_feature @ F_t_a.permute(0,2,1))
                B, L, C = anomaly_map_new.shape 
                H = int(np.sqrt(L))
                anomaly_map_new = F.interpolate(anomaly_map_new.permute(0, 2, 1).view(B,2,H,H),
                                            size = img_size, mode = 'bilinear', align_corners=True)

                anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)[:, 1, :, :]
                anomaly_maps_new.append(anomaly_map_new.cpu().numpy())

            patch_tokens_linear = linearlayer(patch_tokens)
            anomaly_maps_raw = []
            for layer in range(len(patch_tokens_linear)):
                dense_feature = patch_tokens_linear[layer].clone()
                dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                anomaly_map_raw = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings)
                B, L, C = anomaly_map_raw.shape 
                H = int(np.sqrt(L))
                anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                            size = img_size, mode = 'bilinear', align_corners=True)

                anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)[:, 1, :, :]
                anomaly_maps_raw.append(anomaly_map_raw.cpu().numpy())

            anomaly_map_raw = np.mean(anomaly_maps_raw, axis=0)
            anomaly_map_new = np.mean(anomaly_maps_new, axis=0)

            
            results['anomaly_map_raw'].append(anomaly_map_raw)
            results['anomaly_map_new'].append(anomaly_map_new)

            path = items['img_path']
            results['path'].extend(path)

    calcuate_metric(results, obj_list, logger ,alpha = 0.2, args = args)

def calcuate_metric(results, obj_list, logger, alpha = 0.2, args = None):

    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_px_ls = []
    aupro_sp_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    iou_list = []
    iou_list_ls = []
    table_best_the = []

    names_list = np.array(results['cls_names'])
    gts_img = torch.tensor(results['gt_sp']).cpu().numpy()
    imgs_path = np.array(results['path'])

    anomaly_maps_raw = np.concatenate(results['anomaly_map_raw'], axis=0)
    anomaly_maps_new = np.concatenate(results['anomaly_map_new'], axis=0)
    gts_pixel = np.concatenate(results['imgs_masks'], axis=0)
    
    for obj in obj_list: 
        table = []
        table.append(obj)
        can_k = -2000

        object_index = np.where(names_list == obj)[0]
        img_path_list = imgs_path[object_index]
        pr_px_1 = anomaly_maps_raw[object_index,:,:].copy()
        pr_px_2 = anomaly_maps_new[object_index,:,:].copy()
        gt_sp = gts_img[object_index].copy()
        gt_px = gts_pixel[object_index,:,:].copy()
        pr_px =  normalize(gaussian_filter(alpha * pr_px_1 + (1 - alpha) * pr_px_2, sigma=8,axes = (1,2))) 
        pr_sp_1 = np.partition(pr_px_1.reshape(pr_px_1.shape[0],-1), kth=can_k)[:, can_k:]
        pr_sp_2 = np.partition(pr_px_2.reshape(pr_px_2.shape[0],-1), kth=can_k)[:, can_k:]

        pr_sp_tmp = np.mean(pr_sp_1, axis=1) + np.mean(pr_sp_2, axis=1)
        #pr_sp_tmp = np.max(pr_px, axis=(1,2))
        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
        pr_sp = pr_sp_tmp
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())    # pixel-level AUROC
        auroc_sp = roc_auc_score(gt_sp, pr_sp)  # image-level AUROC
        ap_sp = average_precision_score(gt_sp, pr_sp)  # image-level AP
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())  # pixel-level AP

        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])  # image-level F1-max

        # aupr
        #aupro_sp = auc(recalls, precisions)  # image-level PRO
        aupro_sp = 0
        
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls+ 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])   # pixel-level F1-max 
        iou = cal_iou(gt_px.ravel(), (pr_px.ravel()>best_threshold))  # mIoU
        iou_list.append(iou)
        print("{}--->  iou:{}   f1-max:{}  threshold:{}".format(obj,iou,f1_px,best_threshold))

        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro_px = cal_pro_score(gt_px, pr_px) # pixel-level AUPRO
        #aupro_px = 0
        

        #----------------------------------start visualization --------------------------#
        print("visualization {}".format(obj))
        
        for i in range(len(img_path_list)):
            cls = img_path_list[i].split('/')[-2]
            filename = img_path_list[i].split('/')[-1]
            save_vis = os.path.join(args.save_path, 'imgs', obj, cls)
            vis_img = vis_img = cv2.resize(cv2.imread(img_path_list[i]), (args.image_size, args.image_size))
            visualization(save_root= save_vis, pic_name=filename, raw_image= vis_img, raw_anomaly_map= np.squeeze(pr_px[i]), raw_gt= np.squeeze(gt_px[i]), the = best_threshold)
        #----------------------------------end visualization --------------------------#

        table.append(str(np.round(auroc_px * 100, decimals=2)))   
        table.append(str(np.round(aupro_px * 100, decimals=2)))
        table.append(str(np.round(ap_px * 100, decimals=2)))

        table.append(str(np.round(f1_px * 100, decimals=2)))
        table.append(str(np.round(iou * 100, decimals=2)))


        table.append(str(np.round(auroc_sp * 100, decimals=2)))
        table.append(str(np.round(aupro_sp * 100, decimals=2)))

        table.append(str(np.round(ap_sp * 100, decimals=2)))
        table.append(str(np.round(f1_sp * 100, decimals=2)))
        table.append(str(np.round(best_threshold, decimals=3)))
        

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_px_ls.append(aupro_px)
        aupro_sp_ls.append(aupro_sp)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)
        iou_list_ls.append(iou)
        table_best_the.append(best_threshold)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=2)),
                     str(np.round(np.mean(aupro_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_px_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(iou_list_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(auroc_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(aupro_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(table_best_the), decimals=3))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou',"auroc_sp","aupro_sp","ap_sp", "f1_sp", "threshold"], tablefmt="pipe")
    logger.info("\n%s", results)
    print(args.checkpoint_path)


import shutil
def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("VCP-CLIP", add_help=True)
    parser.add_argument("--data_path", type=str, default="./dataset/mvisa/data", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/ceshi/zero_shot', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./vcp_weight/train_visa/train_visa.pth', help='weight path of VCP')
    parser.add_argument("--config_path", type=str, default='./models/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default='./pretrained_weight/ViT-L-14-336px.pt', help="weight path of CLIP model")
    # model

    parser.add_argument("--dataset", type=str, default='mvtec', help="testing dataset name, mvtec, visa, or other")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")


    parser.add_argument("--image_size", type=int, default=518, help="image size")

    parser.add_argument("--seed", type=int, default=333, help="random seed")
    parser.add_argument("--prompt_len", type=int, default=2, help="the length of the learnable category vectors r")
    parser.add_argument("--deep_prompt_len", type=int, default=1, help="the length of the learnable text embeddings n ")
    parser.add_argument("--device_id", type=int, default=1, help="GPU id: >=0")
    parser.add_argument("--use_global", default=True, action="store_false", help="Whether to use global visual context in the Pre-VCP module")
    parser.add_argument("--total_d_layer_len", type=int, default= 11, help="number of layers for the text encoder with learnable text embeddings")

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    if "ceshi" in args.save_path:
        move(args.save_path)
    setup_seed(args.seed)
    test(args)