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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from albumentations import ImageCompression, Compose
#URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
         #      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
         #      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

CLASS_NAMES = ['noiseprint']
test_path = "/nas/home/ajaramillo/projects/Try/images"
train_df = pd.read_csv("train.csv")

dataset_path = "/nas/home/ajaramillo/projects/datasets"
#dataset_path = "/nas/public/dataset/korus-realistic-tampering"
test_folder = "dso-dsi"
#test_folder = "Nikon_D90"


class NoiseprintDataset(Dataset):
    def __init__(self, class_name='noiseprint', is_train=True, num_images = 20,
                 resize=256, cropsize=224, model ="uformer"):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.num_images = num_images
        self.model = model
        self.test_image = "/nas/home/ajaramillo/projects/Try/images/people.png"

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

        with open("classes_idx", "rb") as fp:  # Unpickling
            classes_idx = pickle.load(fp)
        self.classes_idx = classes_idx
        self.num_samples_per_class = [len(self.classes_idx[i]) for i in range(len(self.classes_idx))]
        self.classes = [i for i in range(self.num_samples_per_class.__len__())]
        self.known_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 33,34,35,36,37,38,39,40,41,42,43,44]
        self.unknown_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.known_classes_bkrg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.known_classes_ptchs = [33,34,35,36,37,38,39,40,41,42,43,44]
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
                                        #T.CenterCrop(128),
                                        T.ToTensor(),
                                        ])

        self.jpeg = Compose([ImageCompression(quality_lower=98, quality_upper=99, always_apply=True, p=1)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        if self.is_train is True:
            # x = self.x[idx]

            if self.model == "dncnn" or self.model == "restormer":
                x = Image.open(x).convert('L')
            else:
                x = Image.open(x).convert('RGB')
            x = x.crop((0, 0, 0 + self.cropsize*4, 0 + self.cropsize*4))
            x = self.transform_x(x)

        else:
            if self.model == "dncnn" or self.model == "restormer":
                x = Image.open(x).convert('L')
                #w, h = x.size
            else:
                x = Image.open(x).convert('RGB')
                #w, h = x.size

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
            img_1[:, p_posy:p_posy + forgery_size, p_posx:p_posx + forgery_size] = img_2[:,
                                                                                   f_posy:f_posy + forgery_size,
                                                                                   f_posx:f_posx + forgery_size]
            x = img_1

            # ratio = 0.9
            # basewidth = int(x.size[0] * ratio)
            # wpercent = (basewidth / float(x.size[0]))
            # hsize = int((float(x.size[1]) * float(wpercent)))
            # x = x.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            #x = np.array(x)
            #x = self.jpeg(image=x)["image"]
            #x = x[6:-1, 7:-1]
            #x = self.transform_test(x)

        #x = x / 255.
        #x = genNoiseprint(x, modelname="DnCNN", returnnumpy=False)
        #x = x.squeeze(0)
        #x = x.repeat(3, 1, 1)
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = torch.zeros_like(x)
            mask[:, p_posy:p_posy + forgery_size, p_posx:p_posx + forgery_size] = 1
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
                img_fpath_list += [
                    train_df.loc[train_df["id"] == self.classes_idx[self.unknown_classes_bkgr[i]][s], "probe"].values[0]
                    for s in imgs]
            # x.extend(img_fpath_list)
            # img_fpath_list = ["/nas/home/ajaramillo/projects/Try/images/people.png"]
            y.extend([1] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
            x.extend(img_fpath_list)
        else:
            img_fpath_list = []
            for c in self.known_classes:
                samples = random.sample(range(self.num_samples_per_class[c]), self.num_images)
                img_fpath_list += [train_df.loc[train_df["id"] == self.classes_idx[c][s], "probe"].values[0] for s in
                                   samples]
            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        return list(x), list(y), list(mask)

if __name__ == '__main__':
    train_dataset = NoiseprintDataset(is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    for x in tqdm(train_dataloader):
        print(x)



