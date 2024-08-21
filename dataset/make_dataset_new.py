

import numpy as np 
import os 
import shutil 
import cv2
from pycocotools.coco import COCO
import glob

from datasets200.VisA import Visa_dataset
from datasets200.MVTec import Mvtec_dataset




def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)



if __name__ == "__main__":
    
    des_root = "./dataset/mvisa/data/visa"   # generated unified dataset path
    move(des_root)
    id = 0

    Visa_Dataset = Visa_dataset("Your root path/visa")   # original dataset path
    id = Visa_Dataset.make_VAND(binary=True,to_255=True,des_path_root=des_root,id=id) 
    print(id)


    #-------------------------------------------------------------------------------
    

    
    des_root = "./dataset/mvisa/data/mvtec"   # generated unified dataset path
    move(des_root)
    id = 0

    Mvtec_Dataset = Mvtec_dataset("Your root path/mvtec")  #original dataset path
    id = Mvtec_Dataset.make_VAND(binary=True,to_255=True,des_path_root=des_root,id=id)
    print(id)
    

    


    