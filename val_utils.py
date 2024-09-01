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
from utils.extra_utils import get_asd_label_index,get_name_id,label_length,get_asd_label_index_with_id
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
import sklearn.metrics
from utils.tools import reduce_tensor
import shap

import matplotlib.pyplot as plt

@torch.no_grad()
def validate(val_loader, text_labels, model, config,logger,writer,full_text_tokens_set,val_data,epoch):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()

    local_gt = []
    local_pred = []

    with torch.no_grad():

        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = get_asd_label_index_with_id(label_id)
            print("\033[0;31;40m label asd\033[0m")
            print(label_id)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, 1)).cuda()


            for i in range(n):
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)
                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()


                cls_set = model(image=image_input,val=True)

                sig_func=torch.nn.Sigmoid()
                similarity=sig_func(cls_set)
                print("\033[0;31;40msimilarty\033[0m")
                print(similarity)
                tot_similarity += similarity
                print(tot_similarity)



            print("\033[0;31;40m val ind\033[0m")
            print(tot_similarity)

            print("\033[0;31;40m label asd\033[0m")
            print(label_id)

            print("\033[0;31;40m val\033[0m")
            acc1, acc5 = 0, 0
            for i in range(b):
                if label_id[i]==0:
                    local_gt.append(0)
                else:
                    local_gt.append(1)
                local_pred.append(tot_similarity[i].clone().cpu().numpy()[0])


                if tot_similarity[i]>=0.5 and label_id[i]==1 or tot_similarity[i]<0.5 and label_id[i]==0:
                    acc1 += 1
                    print("match")
                else:
                    print("unmatch")

            acc1_meter.update(float(acc1) / b * 100, b)

            # acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
                print("\033[0;31;40macc1=\033[0m" + str(acc1) + " " + str(b))
                print("\033[0;31;40msum=\033[0m" + str(acc1_meter.sum))
                print("\033[0;31;40mcount=\033[0m" + str(acc1_meter.count))




        rank = dist.get_rank()

        print("\033[0;31;40m local auc\033[0m")
        print(local_gt)
        print(local_pred)

        loc_auc_score = sklearn.metrics.roc_auc_score(local_gt, local_pred)
        print("\033[0;31;40m local auc\033[0m")
        print(loc_auc_score)



        def average_tensor(tensor):
            size = float(dist.get_world_size())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= size
            return tensor


        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thersholds = roc_curve(local_gt, local_pred,pos_label=1)
        for i, value in enumerate(thersholds):
            print("%f %f %f" % (fpr[i], tpr[i], value))
        roc_auc = auc(fpr, tpr)
        avg_auc=average_tensor(torch.tensor(roc_auc).to(similarity.device))
        avg_auc=avg_auc.cpu().numpy()
        writer.add_scalar('auc', avg_auc, epoch)
        print("\033[0;31;40m global auc\033[0m")
        print(avg_auc)

    acc1_meter.sync()

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    return acc1_meter.avg



