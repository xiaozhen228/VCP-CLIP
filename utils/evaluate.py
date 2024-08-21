import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score


def evaluate_pre(test_dataloader, model, linearlayer, New_Lan_Embed,prompt_ceshi, device, args, obj_list):
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    id = 0
    for items in test_dataloader:
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, args.features_list)

            # sample
            patch_tokens = linearlayer(patch_tokens)
            class_token = New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings = prompt_ceshi.forward_ensemble(model, class_token, device)

            text_embeddings = text_embeddings.permute(0,2,1)

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
               
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (New_Lan_Embed.prompt_temp.exp() * patch_tokens[layer] @ text_embeddings)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            
            anomaly_map = np.sum(anomaly_maps, axis=0)[0]
            results['anomaly_maps'].append(anomaly_map)
    # metrics
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)

    ap_mean = np.mean(ap_px_ls)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    del results, gt_px, pr_px
    return ap_mean 

def evaluate_post(test_dataloader, model, linearlayer, New_Lan_Embed,Zero_try, prompt_ceshi, device, args, obj_list):
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    Zero_try.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    id = 0
    for items in test_dataloader:
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, args.features_list)

            class_token = New_Lan_Embed.before_extract_feat(patch_tokens,image_features.clone(), use_global = args.use_global)
            text_embeddings = prompt_ceshi.forward_ensemble(model, class_token, device)

            text_embeddings = text_embeddings.permute(0,2,1)

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer][:,1:,:]
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                dense_feature = patch_tokens[layer].detach()
                F_s_a, F_t_a = Zero_try(text_embeddings.permute(0,2,1), dense_feature)
                anomaly_map = (Zero_try.prompt_temp_l1.exp() * dense_feature @ F_t_a.permute(0,2,1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            
            anomaly_map = np.sum(anomaly_maps, axis=0)[0]            
            results['anomaly_maps'].append(anomaly_map)
    # metrics
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])

        gt_px = np.array(gt_px)  
        pr_px = np.array(pr_px)  
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)
    ap_mean = np.mean(ap_px_ls)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    Zero_try.train()
    del results, gt_px, pr_px
    return ap_mean 




def evaluate(test_dataloader, model, linearlayer, New_Lan_Embed,Zero_try, prompt_ceshi, device, args, obj_list):
    model.eval()
    linearlayer.eval()
    New_Lan_Embed.eval()
    Zero_try.eval()

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
    print("---------------------------start evaluation---------------------------")
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
            image_features, patch_tokens = model.encode_image(image, args.features_list)

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
                                            size = args.image_size, mode = 'bilinear', align_corners=True)

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
                                            size = args.image_size, mode = 'bilinear', align_corners=True)

                anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)[:, 1, :, :]
                anomaly_maps_raw.append(anomaly_map_raw.cpu().numpy())

            anomaly_map_raw = np.mean(anomaly_maps_raw, axis=0)
            anomaly_map_new = np.mean(anomaly_maps_new, axis=0)

            
            results['anomaly_map_raw'].append(anomaly_map_raw)
            results['anomaly_map_new'].append(anomaly_map_new)

    # metrics
    ap_px_ls_1 = []
    ap_px_ls_2 = []
    names_list = np.array(results['cls_names'])
    anomaly_maps_raw = np.concatenate(results['anomaly_map_raw'], axis=0)
    anomaly_maps_new = np.concatenate(results['anomaly_map_new'], axis=0)
    gts_pixel = np.concatenate(results['imgs_masks'], axis=0)
    for obj in obj_list:

        object_index = np.where(names_list == obj)[0]

        pr_px_1 = anomaly_maps_raw[object_index,:,:].copy()
        pr_px_2 = anomaly_maps_new[object_index,:,:].copy()

        gt_px = gts_pixel[object_index,:,:].copy()

        ap_px_1 = average_precision_score(gt_px.ravel(), pr_px_1.ravel())
        ap_px_2 = average_precision_score(gt_px.ravel(), pr_px_2.ravel())

        ap_px_ls_1.append(ap_px_1)
        ap_px_ls_2.append(ap_px_2)

    ap_mean_1 = np.mean(ap_px_ls_1)
    ap_mean_2 = np.mean(ap_px_ls_2)
    model.train()
    linearlayer.train()
    New_Lan_Embed.train()
    Zero_try.train()
    del results, gt_px, pr_px_1, pr_px_2
    return ap_mean_1, ap_mean_2