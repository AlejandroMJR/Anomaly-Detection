import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.patched_dataset as noiseprint_dataset
from model import DnCNN, Uformer, Restormer
from utils import architectures
import time
import torch.nn as nn

# device setup
use_cuda = torch.cuda.is_available()
device = 'cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu'


# device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--save_path', type=str, default='./noiseprint_result')
    parser.add_argument('--arch', type=str, choices=['restormer', 'resnet18', 'wide_resnet50_2'
        , 'dncnn', 'uformer', "efficient"], default='restormer')
    return parser.parse_args()


def main():
    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        ps = 224
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        ps = 224
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == "restormer" or args.arch == "restormer2":
        ps = 48
        model = Restormer()
        state = torch.load("/nas/home/ajaramillo/projects/Try/weigths/net-Restormer/bestval.pth",
                           map_location=lambda storage, loc: storage.cuda(0))["net"]
        # Remove "module." prefix from keys
        net_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove the "module." prefix
                net_state[new_key] = value
            else:
                net_state[key] = value
        model.load_state_dict(net_state)
        t_d = 1
        d = 1
    elif args.arch == "dncnn":
        ps = 48
        model = DnCNN()
        model.load_state_dict(
            torch.load("datasets/bestval.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])
        t_d = 64*3
        d = 64*3
    elif args.arch == "efficient":
        ps = 224
        weights_path = '/nas/home/ajaramillo/projects/noiseprint_classifier/weights_rgb/net-EfficientNetB0_lr-0.001_aug-None_aug_p-0_batch_size-12_num_classes-64/bestval.pth'
        network_class = getattr(architectures, "EfficientNetB0")
        model = network_class(n_classes=64, pretrained=True)
        model.load_state_dict(torch.load(weights_path)["net"])
        model = model
        t_d = 208
        d = 208
    elif args.arch == "uformer":
        ps = 128
        t_d = 1
        d = 1
        model = Uformer(img_size=ps, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff',
                        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=3)
        model.load_state_dict(
            torch.load("datasets/bestval_uformer.pth", map_location=lambda storage, loc: storage.cuda(0))["net"])

    model.to(device)
    model.eval()

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    # model.layer1[-1].register_forward_hook(hook)
    # model.layer2[-1].register_forward_hook(hook)
    # model.layer3[-1].register_forward_hook(hook)

    if args.arch == "dncnn":
        model.model[-7].register_forward_hook(hook)
        model.model[-5].register_forward_hook(hook)
        model.model[-3].register_forward_hook(hook)
    elif args.arch == "restormer2":
        # model.decoder_level1.register_forward_hook(hook)
        model.refinement.register_forward_hook(hook)
        model.output.register_forward_hook(hook)
    elif args.arch == "restormer":
        # model.decoder_level1.register_forward_hook(hook)
        # model.refinement.register_forward_hook(hook)
        model.output.register_forward_hook(hook)
    elif args.arch == "efficient":
        model.efficientnet._blocks[0].register_forward_hook(hook)
        model.efficientnet._blocks[5].register_forward_hook(hook)
        model.efficientnet._blocks[10].register_forward_hook(hook)
        # model._conv_head.register_forward_hook(hook)
        print("check")
    else:
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # fig_img_rocauc = ax[0]
    # fig_pixel_rocauc = ax[1]

    # total_roc_auc = []
    # total_pixel_roc_auc = []

    for class_name in noiseprint_dataset.CLASS_NAMES:

        train_dataset = noiseprint_dataset.NoiseprintDataset(cropsize=ps, class_name=class_name, is_train=True,
                                                             model=args.arch)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
        test_dataset = noiseprint_dataset.NoiseprintDataset(cropsize=ps, class_name=class_name, is_train=False,
                                                            model=args.arch)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        if args.arch == "dncnn" or args.arch == "efficient" or args.arch == "resnet18" or args.arch == "wide_resnet50_2":
            train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        elif args.arch == "restormer2":
            train_outputs = OrderedDict([('layer1', []), ('layer2', [])])
            test_outputs = OrderedDict([('layer1', []), ('layer2', [])])
        else:
            train_outputs = OrderedDict([('layer1', [])])
            test_outputs = OrderedDict([('layer1', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    posh = random.randint(1, int((x.shape[-2] - ps) / 8)) - 1
                    posw = random.randint(1, int((x.shape[-1] - ps) / 8)) - 1
                    # posh = random.randint(8, x.shape[-2] - ps - 8)
                    # posw = random.randint(8, x.shape[-1] - ps - 8)
                    train_outputs[k].append(
                        v[:, :, 8 + posh * 8:8 + posh * 8 + ps, 8 + posw * 8:8 + posw * 8 + ps].cpu().detach())
                    # train_outputs[k].append(
                    #     v[:, :, posh : posh  + ps,  posw : posw  + ps].cpu().detach())
                    #train_outputs[k].append(v[:,:,posh:posh+ps,posw:posw+ps].cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat

            if args.arch == "restormer":
                embedding_vectors = train_outputs['layer1']
                # for layer_name in ['layer2']:
                #    embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            elif args.arch == "restormer2":
                embedding_vectors = train_outputs['layer1']
                for layer_name in [ 'layer2']:
                   embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
            else:
                embedding_vectors = train_outputs['layer1']
                for layer_name in ['layer2', 'layer3']:
                    embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0)
            cov = torch.zeros(C, C, H * W)
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f, protocol=4)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []
        all_rec_maps = []
        # extract test set features
        ##########PASS TRHOUGH DATALOADER ALL THE PATCHES FROM THE IMAGE, MAKE A FOR INSIDE THE TQDM
        ##########TO PROCESS PAATCH BY PATCH, RECONSTRUCT THE IMAGE AND THEN GET THE HEATMAP

        pad = 48
        reflection_pad = nn.ReflectionPad2d(padding=pad)
        count = 0
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            padded_image = reflection_pad(x)


            if args.arch == "restormer":
                test_outputs = OrderedDict([('layer1', [])])
            else:
                test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction

            h = x.shape[2]
            w = x.shape[3]
            nh = h // ps
            nw = w // ps
            if h % ps != 0:
                nh = nh + 1
            if w % ps != 0:
                nw = nw + 1

            # padding_height = (ps - x_padded.shape[2] % ps) % ps
            # padding_width = (ps - x_padded.shape[3] % ps) % ps

            # padded_image = F.pad(x_padded,
            #                      (padding_width // 2, padding_width // 2, padding_height // 2, padding_height // 2),
            #                      mode='circular')

            for i in range(nh * nw):
                # Calculate patch position in the image
                patch_row = i // nw
                patch_col = i % nw

                # Calculate patch coordinates
                top = patch_row * ps
                left = patch_col * ps
                bottom = top + ps
                right = left + ps

                top += pad
                left += pad
                bottom += pad
                right += pad

                excess = 16
                # Extract the patch from the image
                patch = padded_image[:, :, top - excess:bottom + excess, left - excess:right + excess]

                with torch.no_grad():
                    _ = model(patch.to(device))
                # get intermediate layer outputs
                for k, v in zip(test_outputs.keys(), outputs):
                    test_outputs[k].append(v[:, :, excess:excess + ps, excess:excess + ps].cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)

            print("check")
            # Embedding concat
            if args.arch == "restormer":
                embedding_vectors = test_outputs['layer1']
                # for layer_name in [ 'layer2']:
                #    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
            elif args.arch == "restormer2":
                embedding_vectors = test_outputs['layer1']
                for layer_name in ['layer2']:
                   embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
            else:
                embedding_vectors = test_outputs['layer1']
                for layer_name in ['layer2', 'layer3']:
                    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

            # calculate distance matrix
            B, C, H, W = embedding_vectors.size()
            total_patches = nw * nh
            reconstructed_image_temp = np.zeros((C, ps * nh, ps * nw))
            for i in range(total_patches):
                row = i // nw
                col = i % nw
                start_row = row * H
                start_col = col * W
                end_row = start_row + H
                end_col = start_col + W
                reconstructed_image_temp[:, start_row:end_row, start_col:end_col] = embedding_vectors[i]

            stride = ps // 2
            num_patches_height = np.int(np.ceil((h - ps) / stride) + 1)
            num_patches_width = np.int(np.ceil((w - ps) / stride) + 1)
            patches = []

            # Extract patches with overlap
            for i in range(num_patches_height):
                for j in range(num_patches_width):
                    start_h = i * stride
                    start_w = j * stride
                    end_h = start_h + ps
                    end_w = start_w + ps
                    patch = reconstructed_image_temp[:, start_h:end_h, start_w:end_w]
                    patches.append(patch)
            patches = np.array(patches)
            patches = torch.tensor(patches).float()
            B, C, H, W = patches.size()
            embedding_vectors = patches.view(B, C, H * W)
            # dist_list = []
            # for i in range(H * W):
            #     mean = train_outputs[0][:, i]
            #     conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            #     dist = []
            #     for data in embedding_vectors:
            #         dist.append(mahalanobis(data[:, i], mean, conv_inv))
            #     dist_list.append(dist)
            mean = train_outputs[0]
            cov_inv = torch.linalg.inv(train_outputs[1].permute(2, 0, 1))
            dist_list = []
            B = 1
            for data in embedding_vectors:
                dist_list.append(efficient_mahalanobis(data, mean, cov_inv, (B, H, W, C)))
            dist_list = torch.cat(dist_list, 0)

            # dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

            # upsample
            # dist_list = torch.tensor(dist_list)

            if args.arch == "restormer2" or args.arch == "restormer" or args.arch == "dncnn":
                score_map = dist_list.numpy()

            else:
                score_map = F.interpolate(dist_list.unsqueeze(1), size=ps, mode='bilinear',
                                          align_corners=False).squeeze().numpy()
            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            # Calculate the number of patches in each dimension
            n_h = math.ceil(h / ps)
            n_w = math.ceil(w / ps)

            # Initialize the output tensor for the reconstructed image
            reconstructed_map = np.zeros((n_h * ps, n_w * ps))

            # Fill the reconstructed image tensor with the patches
            # for i in range(n_h):
            #     for j in range(n_w):
            #         patch = score_map[i * n_w + j]
            #         reconstructed_map[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = patch
            count_image = np.zeros((n_h * ps, n_w * ps))
            for i in range(num_patches_height):
                for j in range(num_patches_width):
                    start_h = i * stride
                    start_w = j * stride
                    end_h = start_h + ps
                    end_w = start_w + ps
                    patch = score_map[i * num_patches_width + j]
                    reconstructed_map[start_h:end_h, start_w:end_w] += patch
                    count_image[start_h:end_h, start_w:end_w] += 1

            reconstructed_map /= count_image
            # Crop the reconstructed image to the original size
            reconstructed_map = reconstructed_map[:h, :w]
            #filename = os.path.join("Mandrake", f"image_{count}.png")

            count+=1
            reconstructed_map = gaussian_filter(reconstructed_map, sigma=4)
            plt.imshow(reconstructed_map), plt.colorbar(), plt.show()
            all_rec_maps.append(reconstructed_map)
            plt.close()
            print("check")
        # Normalization
        scores = np.zeros_like(all_rec_maps)
        all_rec_maps = np.array(all_rec_maps)
        for i in range(len(all_rec_maps)):
            max_score = all_rec_maps[i].max()
            min_score = all_rec_maps[i].min()
            scores[i] = (all_rec_maps[i] - min_score) / (max_score - min_score)
        # max_score = all_rec_maps.max()
        # min_score = all_rec_maps.min()
        # scores = (all_rec_maps - min_score) / (max_score - min_score)
        # scores = 1-scores
        # calculate image-level ROC AUC score
        # img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        # gt_list = np.asarray(gt_list)
        # fpr, tpr, _ = roc_curve(gt_list, img_scores)
        # img_roc_auc = roc_auc_score(gt_list, img_scores)
        # total_roc_auc.append(img_roc_auc)
        # print('image ROCAUC: %.3f' % (img_roc_auc))
        # fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

        # get optimal threshold
        F1s = []
        AUCs = []
        MMCs = []
        #gt_mask = np.array(gt_mask_list)
        gt_mask = gt_mask_list
        #for i in range(gt_mask.shape[0]):
        for i in range(gt_mask.__len__()):
            precision1, recall1, thresholds1 = precision_recall_curve(gt_mask[i].flatten(), scores[i].flatten())
            a1 = 2 * precision1 * recall1
            b1 = precision1 + recall1
            f11 = np.divide(a1, b1, out=np.zeros_like(a1), where=b1 != 0)
            precision2, recall2, thresholds2 = precision_recall_curve(1 - gt_mask[i].flatten(), scores[i].flatten())
            a2 = 2 * precision2 * recall2
            b2 = precision2 + recall2
            f12 = np.divide(a2, b2, out=np.zeros_like(a2), where=b2 != 0)
            if f11.max() > f12.max():
                F1s.append(f11.max())
            else:
                F1s.append(f12.max())
            # threshold = thresholds[np.argmax(f1)]

            # calculate per-pixel level ROCAUC
            fpr1, tpr1, _ = roc_curve(gt_mask[i].flatten(), scores[i].flatten())
            fpr2, tpr2, _ = roc_curve(1 - gt_mask[i].flatten(), scores[i].flatten())
            p1 = gt_mask[i][gt_mask[i] == 1].size
            n1 = gt_mask[i][gt_mask[i] == 0].size
            p2 = n1
            n2 = p1
            fp1, tp1 = fpr1 * n1, tpr1 * p1
            fp2, tp2 = fpr2 * n2, tpr2 * p2
            tn1 = n1 - fp1
            tn2 = n2 - fp2
            fn1 = p1 - tp1
            fn2 = p2 - tp2
            epsilon = 1e-10
            mmc1 = (tp1 * tn1 - fp1 * fn1) / (
                np.sqrt((tp1 + fp1 + epsilon) * (tp1 + fn1 + epsilon) * (tn1 + fp1 + epsilon) * (tn1 + fn1 + epsilon)))
            mmc2 = (tp2 * tn2 - fp2 * fn2) / (
                np.sqrt((tp2 + fp2 + epsilon) * (tp2 + fn2 + epsilon) * (tn2 + fp2 + epsilon) * (tn2 + fn2 + epsilon)))
            if mmc1.max() > mmc2.max():
                MMCs.append(mmc1.max())
            else:
                MMCs.append(mmc2.max())

            per_pixel_rocauc1 = roc_auc_score(gt_mask[i].flatten(), scores[i].flatten())
            # total_pixel_roc_auc.append(per_pixel_rocauc)
            per_pixel_rocauc2 = roc_auc_score(1 - gt_mask[i].flatten(), scores[i].flatten())
            if per_pixel_rocauc1 > per_pixel_rocauc2:
                AUCs.append(per_pixel_rocauc1)
            else:
                AUCs.append(per_pixel_rocauc2)
        AUCs = np.asarray(AUCs)
        F1s = np.asarray(F1s)
        MMCs = np.asarray(MMCs)
        print('pixel ROCAUC: %.3f' % (AUCs.mean()))
        print('pixel F1s: %.3f' % (F1s.mean()))
        print('pixel MMCs: %.3f' % (MMCs.mean()))
        print("finish")
        #
        # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        # os.makedirs(save_dir, exist_ok=True)
        # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    # print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    # fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    # fig_img_rocauc.legend(loc="lower right")
    #
    # print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.legend(loc="lower right")
    #
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def efficient_mahalanobis(
        embedding_vectors: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor, shape: tuple
) -> torch.Tensor:
    """Mahalanobis distance calculator
    https://github.com/scipy/scipy/blob/703a4eb497900bdb805ca9552856672c7ef11d21/scipy/spatial/distance.py#L285
    Args:
        embedding_vectors (torch.Tensor)
        mean (torch.Tensor): patchwise mean for reference feature embeddings
        cov_inv (torch.Tensor): patchwise inverse of covariance matrix for reference feature embeddings
        shape (tuple): input shape of the feature embeddings
    Returns:
        torch.Tensor: distance from the reference distribution
    """
    B, H, W, C = shape
    embedding_vectors = embedding_vectors.unsqueeze(0)
    embedding_vectors = embedding_vectors.permute(0, 2, 1)
    mean = mean.permute(1, 0)
    delta = torch.unsqueeze(embedding_vectors - mean, dim=-1)
    res = torch.matmul(delta.transpose(-1, -2), torch.matmul(cov_inv, delta))
    dist_list = torch.squeeze(torch.sqrt(res))
    dist_list = dist_list.view(B, H, W)
    return dist_list


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    st = time.time()
    main()
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
