import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import  json 
import argparse 
import numpy as np 
import random 
import os 
import torch 
from torch import nn 
from torch.nn import functional as F 
import torchvision.transforms as transforms 
import logging 
from models.model_CLIP import Load_CLIP, tokenize
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from utils.dataset import Makedataset, Split_Product
from utils.loss import FocalLoss, BinaryDiceLoss
from models.prompt_ensemble import Prompt_Ensemble
from tqdm import tqdm
from models.pre_vcp import Context_Prompting
from models.post_vcp import Zero_Parameter
from utils.evaluate import evaluate_pre, evaluate_post
import copy

def _load_stages(model, params, exclude_key=None):  # Load the weights of learnable prompts.
    for n, m in model.named_parameters():
        if exclude_key and exclude_key in n:
            assert m.data.size() == params[n].data.size()
            m.data = params[n].data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


class LinearLayer(nn.Module): # linear layers used for mapping patch-level features.
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        assert 'ViT' in model
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
    def forward(self, tokens):
        tokens_list = []
        for i in range(len(tokens)):
            tokens_list.append(self.fc[i](tokens[i][:, 1:, :]))
        return tokens_list

def _freeze_stages(model, exclude_key=None):  # Freeze all parameters except for the learnable prompts. (All the parameters that need to be trained in this code have "prompt" in their names.)
    """Freeze stages param and norm stats."""
    parameter_prompt_dict = {}
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:  
                    m.requires_grad = False
                else:
                    parameter_prompt_dict[n] = m
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count>0:  
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False
    return parameter_prompt_dict


def train(args):

    image_size = args.image_size 
    epochs = args.epoch
    tokenizer = tokenize
    learning_rate = args.learning_rate 
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path,"result.txt")
    features_list  = args.features_list 
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    
    #---------------- start writing logs ----------------------#
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_hander  = logging.FileHandler(log_path, mode = 'a+')
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    console_hander = logging.StreamHandler()
    console_hander.setFormatter(formatter)
    logger.addHandler(console_hander)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args,arg)}')
    #---------------- end writing logs ----------------------#


    model , preprocess_train , preprocess_test = Load_CLIP(image_size, args.pretrained_path , device=device, deep_prompt_len = args.deep_prompt_len, total_d_layer_len = args.total_d_layer_len)
    model.to(device)
    model_optim = _freeze_stages(model, "prompt")
    
    Make_dataset = Makedataset(train_data_path = args.train_data_path , preprocess_test = preprocess_test, mode = "train", 
                               train_mode = "zero", image_size = args.image_size  , aug = args.aug_rate)

    Make_dataset_val = Makedataset(train_data_path = args.val_data_path , preprocess_test = preprocess_test, mode = "test", 
                               train_mode = "zero", image_size = args.image_size  , aug = -1)
    
    Product_groups = Split_Product(args.dataset)  # Our code supports using only a subset of products for auxiliary training or validating model performance




    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)
    trainable_layer.train()

    New_Lan_Embed = Context_Prompting(model_configs, cla_len = args.prompt_len).to(device) # Generate global visual context in pre-VCP module
    New_Lan_Embed.train()
    prompt_pre = Prompt_Ensemble(args.prompt_len,tokenizer)  # Generate the initial text embeddings

    Zero_try = Zero_Parameter(dim_v = model_configs["vision_cfg"]['width'], dim_t =  model_configs['text_cfg']['width'], dim_out= model_configs["vision_cfg"]['width']).to(device)
    Zero_try.train()   # Further update text embeddings in the post-VCP module using detailed patch-level features


    parameter_prompt_list = []
    for n, m in New_Lan_Embed.named_parameters():
        if n != "prompt_temp": #prompt_temp is learnable temperature coefficient $\tau_1$
            parameter_prompt_list.append(m)
        else:
            print(n)

    group1 = []
    group2 = []
    for n, m in Zero_try.named_parameters():
        if n not in ["prompt_temp_l1"]: # "prompt_temp_l1 is learnable temperature coefficient $\tau_2$
            group1.append(m)
        else:
            group2.append(m)
            print(n)


    parameter_model_prompt_list = [value for key,value in model_optim.items()]

    lr_group1 = list(trainable_layer.parameters()) +  parameter_prompt_list +  parameter_model_prompt_list
    lr_group2 = [New_Lan_Embed.prompt_temp] 

    optimizer = torch.optim.Adam([{'params': group1 + lr_group1, 'lr': args.learning_rate}, {'params': group2 + lr_group2, 'lr':0.01}], lr = learning_rate, betas = (0.5 , 0.999))  
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()



    for group_id in range(len(Product_groups)):
        if group_id not in args.group_id_list:
            continue
        logger.info(f"pre:{Product_groups[group_id]}")
        pre_dataloader, pre_obj_list = Make_dataset.mask_dataset(name = args.dataset, product_list= Product_groups[group_id]["pre"],batchsize=args.batch_size, shuf= True)
        post_dataloader, post_obj_list = Make_dataset.mask_dataset(name = args.dataset, product_list= Product_groups[group_id]["post"], batchsize= args.batch_size, shuf= True)

        #Select the validation set based on the auxiliary training dataset to evaluate the model's zero-shot anomaly segmentation performance during the training process.
        if args.dataset == "mvtec":  
            val_product_list  = ["chewinggum", "cashew", "pipe_fryum","capsules", "candle"] 
            val_post_dataloader, val_obj_list_post = Make_dataset_val.mask_dataset(name = "visa", product_list= val_product_list, batchsize=1, shuf= False)
        else:
            val_product_list  = ["bottle", "hazelnut", "wood", "zipper", "leather"] 
            val_post_dataloader, val_obj_list_post = Make_dataset_val.mask_dataset(name = "mvtec", product_list= val_product_list, batchsize=1, shuf= False)

        if args.resume_checkpoint_path is not None: # Resume training from a checkpoint
            checkpoint = torch.load(args.resume_checkpoint_path, map_location= device)
            trainable_layer.load_state_dict(checkpoint["trainable_linearlayer"])
            Zero_try.load_state_dict(checkpoint["Zero_try"])
            New_Lan_Embed.load_state_dict(checkpoint["New_Lan_Embed"])
            _load_stages(model, checkpoint, "prompt")


        ap_base_max = 0
        ap_new_max = 0
        
        for epoch in range(epochs):
            loss_list = []
            loss_raw_list = []
            loss_new_list = []
            idx = 0
            post_dataloader = tqdm(post_dataloader)
            for items in post_dataloader:
                idx += 1
                image = items['img'].to(device)
                cls_name = items['cls_name']
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        image_features, patch_tokens = model.encode_image(image, features_list)
                    class_token = New_Lan_Embed.before_extract_feat(patch_tokens,image_features, use_global = args.use_global)
                    text_embeddings = prompt_pre.forward_ensemble(model, class_token, device)

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
                                                    size = image_size, mode = 'bilinear', align_corners=True)

                        anomaly_map_new = torch.softmax(anomaly_map_new, dim =1)
                        anomaly_maps_new.append(anomaly_map_new)
                    
                    patch_tokens_linear = trainable_layer(patch_tokens)
                    anomaly_maps_raw = []
                    for layer in range(len(patch_tokens_linear)):
                        dense_feature = patch_tokens_linear[layer].clone()
                        dense_feature = dense_feature /  dense_feature.norm(dim=-1, keepdim = True)
                        anomaly_map_raw = (New_Lan_Embed.prompt_temp.exp() * dense_feature @ text_embeddings)
                        B, L, C = anomaly_map_raw.shape 
                        H = int(np.sqrt(L))
                        anomaly_map_raw = F.interpolate(anomaly_map_raw.permute(0, 2, 1).view(B,2,H,H),
                                                    size = image_size, mode = 'bilinear', align_corners=True)

                        anomaly_map_raw = torch.softmax(anomaly_map_raw, dim =1)
                        anomaly_maps_raw.append(anomaly_map_raw)


                gt = items['img_mask'].squeeze().to(device)
                gt[gt > 0.5], gt[gt< 0.5] = 1, 0
                loss_new = 0
                for num in range(len(anomaly_maps_new)):
                    loss_new += loss_focal(anomaly_maps_new[num], gt)
                    loss_new += loss_dice(anomaly_maps_new[num][:, 1, :, :], gt)

                loss_base = 0
                for num in range(len(anomaly_maps_raw)):
                    loss_base += loss_focal(anomaly_maps_raw[num], gt)
                    loss_base += loss_dice(anomaly_maps_raw[num][:, 1, :, :], gt)

                loss = loss_new + loss_base
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                loss_raw_list.append(loss_base.item())
                loss_new_list.append(loss_new.item())
                print(loss.item(), loss_base.item(),loss_new.item())
                del patch_tokens_linear, patch_tokens, dense_feature, anomaly_map_raw, anomaly_map_new, loss, loss_new, loss_base
                torch.cuda.empty_cache()

            #-------------------------------Start Evaluation ------------------------------------#
            ap_base = evaluate_pre(val_post_dataloader, model, trainable_layer, New_Lan_Embed, prompt_pre, device, args, val_obj_list_post)
            ap_new = evaluate_post(val_post_dataloader, model, trainable_layer, New_Lan_Embed, Zero_try, prompt_pre, device, args, val_obj_list_post)   
            if ap_new > ap_new_max:
                logger.info('epoch [{}/{}], group_id: {} ap_new_max update:{:.4f}, ap_base_max:{:.4f}'.format(epoch + 1, epochs, group_id, ap_new, ap_base))
                ap_new_max = ap_new
            if ap_base > ap_base_max:
                ap_base_max = ap_base
                logger.info('epoch [{}/{}], group_id: {} ap_new_max :{:.4f}, ap_base_max update:{:.4f}'.format(epoch + 1, epochs, group_id, ap_new, ap_base))

            if (epoch + 1) % args.print_freq == 0:
                logger.info('epoch [{}/{}], group_id: {} loss:{:.4f}  loss_base:{:.4f}   loss_new:{:.4f}  ap_new:{:.4f}  ap_base:{:.4f}'.format(epoch + 1, epochs, group_id, np.mean(loss_list),np.mean(loss_raw_list),np.mean(loss_new_list), ap_new, ap_base))
            #---------------------------- End evaluation ----------------------------------------#
            # save model
            if (epoch + 1) % args.save_freq == 0:
                ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '_group_id_' + str(group_id) + '.pth')
                save_dict = {'trainable_linearlayer': trainable_layer.state_dict(), 'New_Lan_Embed': New_Lan_Embed.state_dict(), "Zero_try":Zero_try.state_dict()}
                save_dict.update(model_optim)
                torch.save(save_dict, ckp_path)


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser("VCP_CLIP", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./dataset/mvisa/data", help="train dataset path")
    parser.add_argument("--val_data_path", type=str, default="./dataset/mvisa/data", help="val dataset path")
    parser.add_argument("--save_path", type=str, default='./my_exps/train_visa', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./models/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="weight path of CLIP model")
    parser.add_argument("--resume_checkpoint_path", type=str, default= None, help="resume_checkpoint_path")

    # model
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name, mvtec, visa, or other")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # training
    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00004, help="learning rate")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="augmentation rate")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--device_id", type=int, default=1, help="GPU id: >=0")
    parser.add_argument("--seed", type=int, default= 333, help="random seed")
    parser.add_argument("--group_id_list", type=int, nargs="+", default=[2], help="default use all products on the mvtec or visa datasets")

    # hyper-parameter
    parser.add_argument("--prompt_len", type=int, default=2, help="the length of the learnable category vectors r")
    parser.add_argument("--deep_prompt_len", type=int, default=1, help="the length of the learnable text embeddings n ")
    parser.add_argument("--use_global", default=True, action="store_false", help="Whether to use global visual context in the Pre-VCP module")
    parser.add_argument("--total_d_layer_len", type=int, default= 11, help="number of layers for the text encoder with learnable text embeddings")
    
    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    setup_seed(args.seed)
    train(args)
    