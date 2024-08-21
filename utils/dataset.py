import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import copy
import albumentations as A
import torchvision.transforms as transforms 

def Split_Product(dataset_name):
	if dataset_name == "visa":
		group1 = ["pcb4", "pipe_fryum", "cashew", "macaroni1"]
		group2 = ["pcb2", "chewinggum", "fryum", "candle"]
		group3 = ["pcb1", "capsules", "pcb3", "macaroni2"]
		group1 = group1 + group2 + group3
		group = [dict(pre=group1+group2, post= group3), dict(pre = group1 + group3, post =group2), dict(pre = group2 + group3, post =group1)]
	
	if dataset_name == "mvtec":

		group1 = ["transistor", "capsule", "metal_nut", "screw", "cable"]
		group2 = ["grid", "wood", "carpet", "toothbrush", "leather"]
		group3 = ["bottle", "tile", "hazelnut", "pill", "zipper"]

		group1 = group1 + group2 + group3

		group = [dict(pre=group1+group2, post= group3), dict(pre = group1 + group3, post =group2), dict(pre = group2 + group3, post =group1)]
	return group

		


class Makedataset():
	def __init__(self, train_data_path, preprocess_test, mode, train_mode, image_size = 518, aug = -1):
		self.train_data_path= train_data_path
		self.preprocess_test = preprocess_test
		self.mode = mode
		self.train_mode = train_mode
		self.target_transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.CenterCrop(image_size),transforms.ToTensor()])
		self.aug_rate = aug
		self.shuf = True if self.mode == "train" else False
		

	def mask_dataset(self, name, product_list, batchsize, shuf = True):
		if name == "mvtec" and self.train_mode == "zero":
			dataset = MVTecDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform,
								  aug_rate=self.aug_rate, mode =self.mode, train_mode=self.train_mode, product_list= product_list, dataset="mvtec")
		elif name == "visa" and self.train_mode == "zero":
			dataset = VisaDataset(root=self.train_data_path, transform=self.preprocess_test, target_transform=self.target_transform, 
											   mode =self.mode, train_mode=self.train_mode, product_list= product_list, dataset= "visa")
		obj_list = dataset.get_cls_names()
		
		dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = shuf)
		
		return dataloader, obj_list
		

			



def merge(di1, di2):
	he_bing = copy.deepcopy(di1)
	for di2_key in di2.keys():
		if len(di2[di2_key]) > 1:
			if di2_key not in di1.keys():
				he_bing[di2_key] = di2[di2_key]
			else:
				he_bing[di2_key].extend(di2[di2_key])
	return he_bing


class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='train', train_mode = 'zero', aug = True, product_list = None, dataset = "visa"):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug = aug
		self.mode = mode 
		self.train_mode = train_mode

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]

		self.img_trans = A.Compose([
			A.Rotate(limit=30, p=0.5),
			A.RandomRotate90( p = 0.5),
			A.RandomBrightnessContrast(p=0.5),
			A.GaussNoise(p=0.5),
			A.OneOf([
				A.Blur(blur_limit=3, p=0.5),
				A.ColorJitter(p=0.5),
				A.GaussianBlur(p=0.5),
			], p=0.5)
		], is_check_shapes=False)

		self.cls_names = list(meta_info.keys())
		
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def Trans( self, img , img_mask):
		img_mask = np.array(img_mask)
		img = np.array(img)[:, :, ::-1]
		augmentations = self.img_trans(mask=img_mask, image=img)
		img = augmentations["image"][:, :, ::-1]
		img_mask = augmentations["mask"]
		img = Image.fromarray(img.astype(np.uint8))
		img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
		return img, img_mask
	
	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		if self.mode == "train" and self.aug == True:
			
			img_mask = np.array(img_mask)
			img = np.array(img)
			augmentations = self.img_trans(mask=img_mask, image=img)
			img = augmentations["image"]
			img_mask = augmentations["mask"]
			img = Image.fromarray(img)
			img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
			
			#img, img_mask = self.Trans(img, img_mask)
		
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='train', train_mode = 'zero', aug = True, dataset = "mvtec", product_list = None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate
		self.mode = mode
		self.train_mode =  train_mode
		self.aug = aug
		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]
		
		self.img_trans = A.Compose([
			A.Rotate(limit=30, p=0.5),
			A.RandomRotate90( p = 0.5),
			A.RandomBrightnessContrast(p=0.5),
			A.GaussNoise(p=0.5),
			A.OneOf([
				A.Blur(blur_limit=3, p=0.5),
				A.ColorJitter(p=0.5),
				A.GaussianBlur(p=0.5),
			], p=0.5)
		], is_check_shapes=False)


		self.cls_names = list(meta_info.keys())
				
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def Trans( self, img , img_mask):
		img_mask = np.array(img_mask)
		img = np.array(img)[:, :, ::-1]
		augmentations = self.img_trans(mask=img_mask, image=img)
		img = augmentations["image"][:, :, ::-1]
		img_mask = augmentations["mask"]
		img = Image.fromarray(img.astype(np.uint8))
		img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
		return img, img_mask
	
	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, "mvtec",cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			#random_defect = random.choice(defect)
			random_defect = "anomaly"
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, "mvtec",cls_name, 'ground_truth', random_defect, random_file.replace("bmp", "png"))
			assert (os.path.exists(img_path))
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				assert os.path.exists(mask_path)
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		random_number = random.random()
		if random_number < self.aug_rate and self.mode == "train":
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		if self.mode == "train" and self.aug == True:
			
			img_mask = np.array(img_mask)
			img = np.array(img)
			augmentations = self.img_trans(mask=img_mask, image=img)
			img = augmentations["image"]
			img_mask = augmentations["mask"]
			img = Image.fromarray(img)
			img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
			
			#img, img_mask = self.Trans(img, img_mask)
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class OtherDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', train_mode = 'zero', aug = True, product_list = None, dataset = None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug = aug
		self.mode = mode 
		self.train_mode = train_mode

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_{dataset}.json', 'r'))
		meta_info = meta_info["test"]
		
		if train_mode == "zero" and product_list is not None:
			keys = meta_info.keys()
			keys = list(keys)
			for product in product_list:
				assert product in keys
			for key in keys:
				if key not in product_list:
					del meta_info[key]

		self.img_trans = A.Compose([
			A.Rotate(limit=30, p=0.5),
			A.RandomRotate90( p = 0.5),
			A.RandomBrightnessContrast(p=0.5),
			A.GaussNoise(p=0.5),
			A.OneOf([
				A.Blur(blur_limit=3, p=0.5),
				A.ColorJitter(p=0.5),
				A.GaussianBlur(p=0.5),
			], p=0.5)
		], is_check_shapes=False)

		self.cls_names = list(meta_info.keys())
		
		for index, cls_name in enumerate(self.cls_names):
			self.data_all.extend(meta_info[cls_name])
		
		#self.data_all = self.data_all[:200]
		self.length = len(self.data_all)

	def Trans( self, img , img_mask):
		img_mask = np.array(img_mask)
		img = np.array(img)[:, :, ::-1]
		augmentations = self.img_trans(mask=img_mask, image=img)
		img = augmentations["image"][:, :, ::-1]
		img_mask = augmentations["mask"]
		img = Image.fromarray(img.astype(np.uint8))
		img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
		return img, img_mask
	
	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		if self.mode == "train" and self.aug == True:
			
			img_mask = np.array(img_mask)
			img = np.array(img)
			augmentations = self.img_trans(mask=img_mask, image=img)
			img = augmentations["image"]
			img_mask = augmentations["mask"]
			img = Image.fromarray(img)
			img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
			
			#img, img_mask = self.Trans(img, img_mask)
		
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}