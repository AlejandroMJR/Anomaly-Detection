import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import DataLoader
from datasets.noiseprint import genNoiseprint, genNoiseprintFromFile
from torchvision.transforms.functional import crop
import numpy as np
import pickle
#URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
         #      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
         #      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

CLASS_NAME = ['noiseprint_controlled']
test_path = "/nas/home/ajaramillo/projects/Try/images"
train_df = pd.read_csv("train.csv")



class NoiseprintDataset(Dataset):
    def __init__(self, class_name='noiseprint_controlled', is_train=True, num_images = 20,
                 resize=256, cropsize=224, model ="dncnn"):
        #assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.num_images = num_images
        self.model = model

        # self.classes_idx = [[] for _ in range(len(train_df.model.unique()))]
        # for i in range(len(train_df)):
        #     for j, klass in enumerate(train_df.model.unique()):
        #         if train_df["model"].values[i] == klass:
        #             self.classes_idx[j].append(train_df["id"].values[i])
        # self.num_samples_per_class = [len(self.classes_idx[i]) for i in range(len(self.classes_idx))]

        # get min size of images
        min_h = 480
        min_w = 640
        # for i in range(len(train_df)):
        #     w = train_df.iloc[i]["width"]
        #     h = train_df.iloc[i]["height"]
        #     if w < min_w:
        #         min_w = w
        #     if h < min_h:
        #         min_h = h
        self.min_h = min_h
        self.min_w = min_w
        # self.classes_idx = [[] for _ in range(len(train_df.model.unique()))]
        # for i in range(len(train_df)):
        #     for j, klass in enumerate(train_df.model.unique()):
        #         if train_df["model"].values[i] == klass:
        #             self.classes_idx[j].append(train_df["id"].values[i])
        with open("classes_idx", "rb") as fp:  # Unpickling
            classes_idx = pickle.load(fp)
        self.classes_idx = classes_idx
        self.num_samples_per_class = [len(self.classes_idx[i]) for i in range(len(self.classes_idx))]
        # self.num_samples_per_class = [355, 657, 237, 227, 763, 634, 217, 168, 217, 207, 216, 171, 235, 188, 259, 159, 253, 163, 210, 312, 287, 254, 266, 428, 271, 216, 236, 155, 154, 169, 281, 363, 172, 372, 224, 567, 188, 925, 630, 2391, 925, 752, 369, 367, 1040, 931, 638, 192, 1019, 854, 589, 687, 645, 541, 725, 405, 275, 275, 275, 275, 275, 275, 275, 275]

        self.classes = [i for i in range(self.num_samples_per_class.__len__())]
        self.known_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        self.unknown_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.known_classes_bkrg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.known_classes_ptchs = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        self.unknown_classes_bkgr = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.unknown_classes_ptchs = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([
                                      #T.Resize(resize, Image.ANTIALIAS),
                                      #T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      #T.Normalize(mean=[0.485, 0.456, 0.406],
                                      #            std=[0.229, 0.224, 0.225])
                                      ])
        self.transform_test = T.Compose([
                                        #T.CenterCrop(cropsize),
                                        T.ToTensor(),
                                        ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        if self.model == "dncnn" or self.model == "restormer":
            x = Image.open(x).convert('L')
        # elif self.model == "efficient" and self.is_train is True:
        #     x = Image.open(x).convert("L")
        #     x = x.crop((8,8,8+self.min_w,8+self.min_h))
        #     x = np.array(x) / 255.
        #     x = genNoiseprint(x, modelname="DnCNN", returnnumpy=False)
        #     #x = x[:, :, 8:8+self.cropsize, 8:8+self.cropsize]
        #     x = x.squeeze(0)
        #     x = x.repeat(3, 1, 1).to("cpu")
        # elif self.model == "efficient" and self.is_train is False:
        #     x = Image.open(x).convert("L")
        #     x = np.array(x) / 255.
        #     x = genNoiseprint(x, modelname="DnCNN", returnnumpy=False)
        #     x = x.squeeze(0)
        #     x = x.repeat(3, 1, 1).to("cpu")
        else:
            x = Image.open(x).convert('RGB')

        if self.is_train is True:
            x = x.crop((8, 8, 8 + self.min_w, 8 + self.min_h))
            x = self.transform_x(x)
        else:
            #x = self.transform_test(x)
            # other_class = 25
            x = x.crop((8, 8, 8 + self.min_w, 8 + self.min_h))
            img_1 = self.transform_test(x)
            other_class = random.choice(self.unknown_classes_ptchs)
            img_2 = random.sample(range(self.num_samples_per_class[other_class]), 1)
            img_2 = train_df.loc[train_df["id"] == self.classes_idx[other_class][img_2[0]], "probe"].values[0]
            img_2 = Image.open(img_2).convert('L')
            img_2 = self.transform_test(img_2)
            forgery_size = 64
            f_posx = random.randint(0, img_2.shape[-1] - forgery_size)
            f_posy = random.randint(0, img_2.shape[-2] - forgery_size)
            p_posx = random.randint(0, img_1.shape[-1] - forgery_size)
            p_posy = random.randint(0, img_1.shape[-2] - forgery_size)
            img_1[:,p_posy:p_posy+forgery_size,p_posx:p_posx+forgery_size] = img_2[:,f_posy:f_posy+forgery_size,f_posx:f_posx+forgery_size]
            x = img_1
        #x = x / 255.
        #x = genNoiseprint(x, modelname="DnCNN", returnnumpy=False)
        #x = x.squeeze(0)
        #x = x.repeat(3, 1, 1)
        if y == 0:
            mask = []
        else:
            mask = torch.zeros_like(x)
            mask[:,p_posy:p_posy+forgery_size,p_posx:p_posx+forgery_size] = 1
            # print("check")
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        if not self.is_train:
            img_fpath_list = []
            for i in range(len(self.unknown_classes_bkgr)):
                imgs = random.sample(range(self.num_samples_per_class[self.unknown_classes_bkgr[i]]), 10)
                img_fpath_list += [train_df.loc[train_df["id"] == self.classes_idx[self.unknown_classes_bkgr[i]][s], "probe"].values[0] for s in imgs]
            # x.extend(img_fpath_list)
            #img_fpath_list = ["/nas/home/ajaramillo/projects/Try/images/people.png"]
            y.extend([1] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
            x.extend(img_fpath_list)
        else:
            img_fpath_list = []
            for c in self.known_classes:
                samples = random.sample(range(self.num_samples_per_class[c]), self.num_images)
                img_fpath_list += [train_df.loc[train_df["id"] == self.classes_idx[c][s], "probe"].values[0] for s in samples]
            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        return list(x), list(y), list(mask)

if __name__ == '__main__':
    train_dataset = NoiseprintDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    for x in tqdm(train_dataloader):
        print(x)



