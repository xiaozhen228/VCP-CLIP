import numpy as np 
import os 
import shutil 
import cv2
import glob
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


class Mvtec_dataset():
    def __init__(self,path_root):
        #self.is_binary = True
        self.is_255 = True
        self.path_root = path_root
        
        self.dataset_name = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
        ]
        
        #self.dataset_name = os.listdir(path_root)
        #self.dataset_name = ["pill"]
    def Binary(self,mask):
        if self.is_255:
            mask[mask<=128] = 0
            mask[mask>128] = 1
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1
        return mask
    
    def make_dirs(self,des_root):
        for data_name in self.dataset_name:
            name = data_name
            dir_list = [os.path.join(des_root,name,"train","good"),os.path.join(des_root,name,"test","good"),os.path.join(des_root,name,"test","anomaly"),os.path.join(des_root,name,"ground_truth","anomaly")]
            for dir in dir_list:
                if not os.path.exists(dir):
                    os.makedirs(dir)

    # train  test  ground_truth
    def make_VAND(self,binary,to_255,des_path_root,id):
        self.make_dirs(des_path_root)
        for data_name in self.dataset_name:
            print("Processing :{}".format(data_name))
            for mode in ["train","test"]:
                data_path = os.path.join(self.path_root,data_name)
                defect_classes = sorted(os.listdir(os.path.join(data_path,mode)))
                if len(defect_classes) == 1 and "good" in defect_classes and mode=="train":
                    data_list = glob.glob(os.path.join(data_path,mode,"good",'*.png'))
                    data_list = sorted(data_list)
                    for i in range(len(data_list)):
                        #name_new = data_name.capitalize()
                        #raw_mask_path = os.path.join(data_path,"ground_truth",os.path.basename(data_list[i]).replace("png","png"))
                        raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                        #raw_mask = cv2.imread(raw_mask_path,cv2.IMREAD_GRAYSCALE)
                        new_class_name = data_name
                        save_img_path = os.path.join(des_path_root,new_class_name,"train","good","mvtec_"+str(id).zfill(6)+".bmp") 
                        #save_img_path = os.path.join(des_path_root,new_class_name,"train","good","mvtec_"+str(id).zfill(6)+".png") 
                        cv2.imwrite(save_img_path,raw_img)
                        id = id + 1
                else:
                    if len(defect_classes) > 1 and mode == "test":
                        for classes in defect_classes:
                            data_list = glob.glob(os.path.join(data_path,mode,classes,'*.png'))
                            data_list = sorted(data_list)
                            if classes == "good":
                                for i in range(len(data_list)):
                                    #name_new = data_name.capitalize()
                                    #raw_mask_path = os.path.join(data_path,"ground_truth",os.path.basename(data_list[i]).replace("png","png"))
                                    raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                                    #raw_mask = cv2.imread(raw_mask_path,cv2.IMREAD_GRAYSCALE)
                                    new_class_name = data_name
                                    save_img_path = os.path.join(des_path_root,new_class_name,"test","good","mvtec_"+"good_"+str(id).zfill(6)+".bmp") 
                                    #save_img_path = os.path.join(des_path_root,new_class_name,"test","good","mvtec_"+"good_"+str(id).zfill(6)+".png") 
                                    cv2.imwrite(save_img_path,raw_img)
                                    id = id + 1
                            else:
                                for i in range(len(data_list)):
                                    
                                    raw_mask_path = os.path.join(data_path,"ground_truth",classes,os.path.basename(data_list[i]).replace(".png","_mask.png"))
                                    raw_img = cv2.imread(data_list[i],cv2.IMREAD_COLOR)
                                    raw_mask = cv2.imread(raw_mask_path,cv2.IMREAD_GRAYSCALE)
                                    if binary:
                                        raw_mask = self.Binary(raw_mask.copy())
                                        if to_255:  
                                            raw_mask =  raw_mask * 255
                                    new_class_name = data_name
                                    save_img_path = os.path.join(des_path_root,new_class_name,"test","anomaly",f"mvtec_{classes}_"+str(id).zfill(6)+".bmp")
                                    #save_img_path = os.path.join(des_path_root,new_class_name,"test","anomaly",f"mvtec_{classes}_"+str(id).zfill(6)+".png")
                                    save_mask_path = os.path.join(des_path_root,new_class_name,"ground_truth","anomaly",f"mvtec_{classes}_"+str(id).zfill(6)+".png")

                                    cv2.imwrite(save_img_path,raw_img)
                                    #cv2.imwrite(save_mask_path,raw_mask, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                                    cv2.imwrite(save_mask_path,raw_mask)
                                    id = id + 1
        print("mvtec finished !")
        return id 
        

                                
                               


