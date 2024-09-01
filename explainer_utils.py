import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
import torch.nn.functional as F
from utils.extra_utils import get_asd_label_index, get_name_id, label_length, get_asd_label_index_with_id
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
import sklearn.metrics
from utils.tools import reduce_tensor
import shap
from clip.model import CLIP
import matplotlib.pyplot as plt
import cv2

#old="result_explain"
#fold="new_result_explain"
@torch.no_grad()
def explain(val_loader, model, config):
    model.eval()



    print("\033[0;31;40mexplain\033[0m")
    print(len(val_loader))
    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label_id = batch_data["label"]
        name_id = get_name_id(label_id)
        label_id = get_asd_label_index_with_id(label_id)
        '''print("\033[0;31;40m label asd\033[0m")
        print(label_id)'''

        print("\033[0;31;40mexplain the result\033[0m")
        print(_image.size())

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)
        print(n)

        for i in range(n):
            single_name_id=name_id[i].cpu().numpy()
            image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
            label_id = label_id.cuda(non_blocking=True)
            image_input = image.cuda(non_blocking=True)
            if config.TRAIN.OPT_LEVEL == 'O2':
                image_input = image_input.half()

            print("\033[0;31;40mname id\033[0m")
            print(name_id)

            exp_input = image_input[0].clone()
            exp_input = torch.unsqueeze(exp_input, dim=0)
            cls_set = model(image=exp_input, val=True)
            sig_func = torch.nn.Sigmoid()
            #baseline_similarity = sig_func(cls_set)
            baseline_similarity = cls_set
            print("\033[0;31;40mbase similarity\033[0m")
            print(baseline_similarity)

            with open("./explain/explain_result" + str(single_name_id) + ".txt", 'a') as f:
                f.write(str(single_name_id)+"###"+"prediction:"+str(baseline_similarity))
                f.write("\n")
            f.close()

            differ_maps = torch.zeros(exp_input.shape)

            step_len = 4

            topx=config.TEST.TOP_X
            top_num=config.TEST.TOP_X_PIXEL_NUM


            print(exp_input.shape)
            with torch.no_grad():
                for image_idx in range(exp_input.shape[1]):
                    print("\033[0;31;40mexplaining image_\033[0m"+str(image_idx))
                    for x_idx in range(int(exp_input.shape[3] / step_len)):
                        for y_idx in range(int(exp_input.shape[4] / step_len)):
                            print(image_idx, x_idx, y_idx)
                            if image_idx>=config.TEST.EXPLAIN_NUM:
                                break
                            
                            masked_imgs = mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len)

                            cls_set = model(image=masked_imgs, val=True)

                            #similarity = sig_func(cls_set)
                            similarity = cls_set

                            mark_influence(single_name_id,differ_maps, similarity - baseline_similarity, image_idx, x_idx, y_idx,
                                           step_len)

                print(differ_maps)
                for image_idx in range(exp_input.shape[1]):
                    bk_img = exp_input.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                    differ_map = differ_maps.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                    org_differ_map = differ_map.copy()
                    result_generate(single_name_id,differ_map,bk_img,image_idx)

                    print("org_map")
                    print(org_differ_map)
                    td_differ_map = org_differ_map.copy()
                    print(td_differ_map)
                    td_differ_map[td_differ_map > 0] = 0
                    td_differ_map = -td_differ_map
                    print(td_differ_map)

                    asd_differ_map = org_differ_map.copy()
                    asd_differ_map[asd_differ_map < 0] = 0

                    print("inp map")
                    print(td_differ_map)
                    top_analyse(single_name_id,td_differ_map,bk_img,image_idx,topx,top_num,"td")
                    top_analyse(single_name_id,asd_differ_map, bk_img, image_idx, topx, top_num, "asd")



def mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len):
    ret_imgs = exp_input.clone()
    #print(ret_imgs.shape)
    ret_imgs[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,
    y_idx * step_len:y_idx * step_len + step_len] = 0
    # print(ret_imgs)
    return ret_imgs


def mark_influence(name_id,differ_maps, similarity_differ, image_idx, x_idx, y_idx, step_len):
    print("\033[0;31;40msimilarity differ\033[0m")

    differ_maps[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,
    y_idx * step_len:y_idx * step_len + step_len] = similarity_differ

    with open("./explain/explain_result"+str(name_id)+".txt", 'a') as f:
        f.write(str(image_idx))
        f.write(",")
        f.write(str(x_idx * step_len+1/2*step_len))
        f.write(",")
        f.write(str(y_idx * step_len + 1 / 2 * step_len))
        f.write("/")
        f.write(str(similarity_differ.detach().cpu().numpy()[0][0]))
        f.write("\n")

    f.close()


def get_normal_color(img):
    max = np.max(img)
    min = np.min(img)
    if max!=min:
        ret = (img - min) / (max - min) * 255
    else:
        ret = (img - min) / 0.000000001 * 255
    return ret

def keep_top_x_unique(matrix,top_x,top_num):
    topx=top_x
    # 将矩阵展平并找到前5个不同的值

    unique_values, counts = np.unique(matrix.flatten(), return_counts=True)
    print("unique_values")
    print(matrix)
    print(unique_values)
    #top5_values = unique_values[np.argsort(counts)][::-1][:topx]
    sorted_list = sorted(unique_values, reverse=True)
    top_values = sorted_list[:topx]
    print("top value")
    print(top_values)

    matching_elements = np.isin(matrix, top_values)
    count = np.count_nonzero(matching_elements)
    print(count)
    if count<2:
        print("top x error")
        exit(1)

    while count>top_num*24*24 and topx>1:
        topx=topx-1
        top_values = sorted_list[:topx]
        matching_elements = np.isin(matrix, top_values)
        count = np.count_nonzero(matching_elements)
        print(count)
    print("top_value_count")
    print(count)

    # 将不是前5个不同值的元素设置为0
    mask = np.isin(matrix, top_values, invert=True)
    #print(mask)
    matrix[mask] = 0

    return matrix



def result_generate(name_id,differ_map,bk_img,image_idx):
    bk_img = get_normal_color(bk_img)

    td_differ_map = differ_map.copy()
    td_differ_map[td_differ_map > 0] = 0
    td_differ_map=-td_differ_map

    asd_differ_map = differ_map.copy()
    asd_differ_map[asd_differ_map < 0] = 0

    result_save(name_id,td_differ_map,bk_img,image_idx,"td")
    result_save(name_id,asd_differ_map,bk_img,image_idx,"asd")






def result_save(name_id,differ_map,bk_img,image_idx,name:str):

    differ_map = get_normal_color(differ_map)
    bk_img = get_normal_color(bk_img)
    #print(bk_img)

    alpha = 0.5  # 第一张图片的权重
    beta = 0.5  # 第二张图片的权重
    gamma = 0  # 亮度参数，设为0即可
    blended_image = cv2.addWeighted(bk_img, alpha, differ_map, beta, gamma)
    #cv2.imwrite('result_explain/bk_img' + str(image_idx) +name+ '.jpg', bk_img)
    #cv2.imwrite('explain/result_img' + str(image_idx) +name+ '.jpg', blended_image[...,::-1])

    differ_map = differ_map[:, :, 0]
    differ_map = cv2.convertScaleAbs(differ_map)
    #print(differ_map.shape)
    heatmap = cv2.applyColorMap(differ_map, cv2.COLORMAP_JET)  # 将cam的结果转成伪彩色图片
    heatmap = np.float32(heatmap) / 255.  # 缩放到[0,1]之间
    bk_img = bk_img[...,::-1] / 255.
    if np.max(bk_img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + bk_img
    cam = cam / np.max(cam)

    path = "explain/hm_"+str(name_id)
    if os.path.exists(path) is False:
        os.makedirs(path)
    cv2.imwrite(path+'/heatmap' + str(image_idx) +name+ '.jpg', np.uint8(255 * cam))


def top_analyse(name_id,differ_map,bk_img,image_idx,topx,top_num,name):

    print("img_idx:"+str(image_idx))
    differ_map = get_normal_color(differ_map)
    bk_img = get_normal_color(bk_img)

    differ_map = keep_top_x_unique(differ_map, topx,top_num)
    differ_map = get_normal_color(differ_map)
    differ_map = differ_map[:, :, 0]
    differ_map = cv2.convertScaleAbs(differ_map)
    heatmap = cv2.applyColorMap(differ_map, cv2.COLORMAP_JET)  # 将cam的结果转成伪彩色图片
    heatmap = np.float32(heatmap) / 255.  # 缩放到[0,1]之间
    bk_img = bk_img / 255.
    #print(bk_img)

    if np.max(bk_img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + bk_img
    cam = cam / np.max(cam)

    path = "explain/hm_" + str(name_id)
    if os.path.exists(path) is False:
        os.makedirs(path)

    if name=="asd":
        cv2.imwrite(path+'/heatmap_top_' + str(image_idx) + name+ '.jpg', np.uint8(255 * cam)[...,::-1])
    else:
        cv2.imwrite(path+'/heatmap_top_' + str(image_idx) + name + '.jpg', np.uint8(255 * cam)[...,::-1])

    path = "explain/bk_" + str(name_id)
    if os.path.exists(path) is False:
        os.makedirs(path)
    cv2.imwrite(path+'/bk_' + str(image_idx) + name + '.jpg', np.uint8(255 *bk_img)[...,::-1])

