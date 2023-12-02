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
#test_folder = "dalle"
#test_folder = "Nikon_D90"


class NoiseprintDataset(Dataset):
    def __init__(self, class_name='noiseprint', is_train=True, num_images = 50,
                 resize=256, cropsize=224, model ="restormer"):
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

        # ratio = 0.9
        # basewidth = int(2048 * ratio)
        # wpercent = (basewidth / float(2048))
        # hsize = int((float(1536) * float(wpercent)))
        self.transform_mask = T.Compose([#T.Resize((1536, 2048), Image.NEAREST),
                                         # T.CenterCrop(cropsize),
                                         T.ToTensor()])
        self.jpeg = Compose([ImageCompression(quality_lower=95, quality_upper=95, always_apply=True, p=1)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        if self.is_train is True:
            # x = self.x[idx]

            if self.model == "dncnn" or self.model == "restormer" or self.model == "restormer2":
                x = Image.open(x).convert('L')
            else:
                x = Image.open(x).convert('RGB')
            x = x.crop((0, 0, 0 + self.cropsize*4, 0 + self.cropsize*4))
            x = self.transform_x(x)

        else:
            if self.model == "dncnn" or self.model == "restormer" or self.model == "restormer2":
                x = Image.open(x).convert('L')
                #w, h = x.size
            else:
                x = Image.open(x).convert('RGB')
                #w, h = x.size

            # nh = h // self.cropsize
            # nw = w // self.cropsize
            # if h % self.cropsize != 0:
            #     nh = nh + 1
            # if w % self.cropsize != 0:
            #     nw = nw + 1
            # # Calculate patch position in the image
            # patch_row = idx // nw
            # patch_col = idx % nw
            #
            # # Calculate patch coordinates
            # top = patch_row * self.cropsize
            # left = patch_col * self.cropsize
            # bottom = top + self.cropsize
            # right = left + self.cropsize
            #
            # # Extract the patch from the image

            #w, h = x.size
            # left = random.randint(1, 7)
            # top = random.randint(1,7)
            # left = 5
            # top = 3
            # x = x.crop((left, top, w-(7-left), h -(7-top)))
            ratio = 0.81
            basewidth = int(x.size[0] * ratio)
            wpercent = (basewidth / float(x.size[0]))
            hsize = int((float(x.size[1]) * float(wpercent)))
            x = x.resize((basewidth, hsize))
            # x = np.array(x)
            # x = self.jpeg(image=x)["image"]
            #x = x.resize((basewidth, hsize))
            #x.save("temp.jpeg", "JPEG")
            #x= Image.open("temp.jpeg").convert("L")
            x = self.transform_test(x)
            #os.remove("temp.jpeg")
            #x = x[:,:1024, :1024]

        #x = x / 255.
        #x = genNoiseprint(x, modelname="DnCNN", returnnumpy=False)
        #x = x.squeeze(0)
        #x = x.repeat(3, 1, 1)
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            if mask.mode == "LA":
                mask = mask.getchannel(0)
            if mask.mode == "RGBA":
                mask = mask.convert("L")
            else:
                mask = mask

            mask = mask.resize((basewidth, hsize))

            mask = self.transform_mask(mask)
            mask[mask > 1e-10] = 1
            #mask = mask[:,:1024, :1024]
            #mask[:, :1024, :1024] = 1
            #mask = mask[:, top:h - (7 - top), left:w - (7 - left)]
            #print("check")
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        if not self.is_train:
            img_dir = os.path.join(dataset_path, test_folder, "DSO-1")
            gt_dir = os.path.join(dataset_path, test_folder, 'DSO-1-Fake-Images-Masks')
            # img_dir = os.path.join(dataset_path, test_folder, "fake")
            # gt_dir = os.path.join(dataset_path, test_folder, 'mask')
            # img_dir = os.path.join(dataset_path, test_folder, "tampered-realistic")
            # gt_dir = os.path.join(dataset_path, test_folder, 'ground-truth')
            #img_dir = "/nas/home/smandelli/Pycharm_projects/semafor_web_images/cerberus/MandrakeDataset"
            files = os.listdir(img_dir)
            sorted_files = sorted(files)

            img_fpath_list = [os.path.join(img_dir, f)
                              for f in sorted_files
                              #if "STAGE" in f
                              if f.startswith('spli')
                              #if f.endswith("1.png")
                              ]
            #img_fpath_list = ["/nas/home/ajaramillo/projects/datasets/dalle/fake/013-2.png"]
            #img_fpath_list = ["/nas/home/ajaramillo/projects/datasets/dso-dsi/DSO-1/splicing-04.png"]
            y.extend([1] * len(img_fpath_list))
            img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
            gt_fpath_list = [os.path.join(gt_dir, img_fname + '.png')
                            for img_fname in img_fname_list]
            # gt_fpath_list = [os.path.join(gt_dir, img_fname[:3] + '-mask.png')
            #                  for img_fname in img_fname_list]
            # gt_fpath_list = [os.path.join(gt_dir,  'mask1024.png')
            #                  for _ in img_fname_list]
            #gt_fpath_list = ["/nas/home/ajaramillo/projects/datasets/dso-dsi/DSO-1-Fake-Images-Masks/splicing-04.png"]

            #gt_fpath_list = img_fpath_list

            mask.extend(gt_fpath_list)
            x.extend(img_fpath_list)
        else:
            # img_ixds = random.sample(range(len(train_df)), self.num_images)
            # img_fpath_list = [train_df.loc[train_df["id"] == i, "probe"].values[0] for i in img_ixds]
            # x.extend(img_fpath_list)
            img_fpath_list = []
            for c in self.classes:
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



