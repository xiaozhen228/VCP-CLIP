import numpy as np 
import os 
import shutil 
import cv2
import pandas as pd
import sys
import random
seed = 228
np.random.seed(seed)
random.seed(seed)


def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


class Visa_dataset():
    def __init__(self,path_root):
        #self.is_binary = False   
        self.is_255 = False  
        self.path_root = path_root
        self.dataset_name =  [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
        ]
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{path_root}/split_csv/1cls.csv', header=0)
        
    def Binary(self,mask, raw_mask_path):
        if self.is_255:
            try:
                assert (np.unique(mask) == np.array([0,255])).all(), f"{raw_mask_path}"
            except AttributeError:
                print(np.unique(mask), raw_mask_path)
                print("mask error")
                sys.exit(1)
            mask[mask<=128] = 0
            mask[mask>128] = 1
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1
        return mask   
    def make_dirs(self,des_root):
        for data_name in self.dataset_name:
            name = self.Change_name(data_name)
            dir_list = [os.path.join(des_root,name,"train","good"),os.path.join(des_root,name,"test","good"),os.path.join(des_root,name,"test","anomaly"),os.path.join(des_root,name,"ground_truth","anomaly")]
            for dir in dir_list:
                if not os.path.exists(dir):
                    os.makedirs(dir)
    
    def Change_name(self,name):
        #name_new = name.capitalize()
        name_new = name
        return name_new


    # train  test  ground_truth
    def make_VAND(self,binary,to_255,des_path_root,id):
        self.make_dirs(des_path_root)
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        for cls_name in self.dataset_name:
            print("Processing :{}".format(cls_name))
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data[2] == 'anomaly' else False
                    if phase == "train":
                        assert not is_abnormal
                        img_path =  os.path.join(self.path_root,data[3])
                        img_name_id = os.path.basename(data[3]).split(".")[0]
                        save_img_path = os.path.join(des_path_root,cls_name,"train","good","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".bmp") 
                        #save_img_path = os.path.join(des_path_root,cls_name,"train","good","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".png") 

                        mask_path = None
                        save_mask_path = None

                    else:
                        if is_abnormal:
                            img_path =  os.path.join(self.path_root,data[3])
                            mask_path = os.path.join(self.path_root,data[4])
                            img_name_id = os.path.basename(data[3]).split(".")[0]
                            save_img_path  = os.path.join(des_path_root,cls_name,"test","anomaly","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".bmp")
                            #save_img_path  = os.path.join(des_path_root,cls_name,"test","anomaly","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".png")
                            save_mask_path = os.path.join(des_path_root,cls_name,"ground_truth","anomaly","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".png")
                        
                        else:
                            img_path =  os.path.join(self.path_root,data[3])
                            img_name_id = os.path.basename(data[3]).split(".")[0]
                            save_img_path  = os.path.join(des_path_root,cls_name,"test","good","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".bmp")
                            #save_img_path  = os.path.join(des_path_root,cls_name,"test","good","visa_"+str(img_name_id)+"_"+str(id).zfill(6)+".png")
                            mask_path = None
                            save_mask_path = None
                    raw_img = cv2.imread(img_path,cv2.IMREAD_COLOR)
                    cv2.imwrite(save_img_path,raw_img)
                    if mask_path is not None:
                        raw_mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
                        raw_mask = self.Binary(raw_mask.copy(),mask_path)
                        if to_255: 
                            raw_mask =  raw_mask * 255
                        #cv2.imwrite(save_mask_path,raw_mask,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                        cv2.imwrite(save_mask_path,raw_mask)
                    id = id + 1
        print("VisAs finished")
        return id