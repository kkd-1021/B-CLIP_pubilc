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

import matplotlib.pyplot as plt


def convert_to_scalar(array_or_scalar):
    if isinstance(array_or_scalar, np.ndarray) and array_or_scalar.size == 1:
        return array_or_scalar.item()
    else:
        return array_or_scalar

class Pred_storage():
    def __init__(self):
        super().__init__()
        self.person_pred_list=[]

    def add_pred(self,name_id, similarity,gt_label_index):
        name_id=name_id.cpu().numpy()
        similarity=similarity.cpu().numpy()
        gt_label_index=gt_label_index.cpu().numpy()

        name_id=convert_to_scalar(name_id)
        similarity=convert_to_scalar(similarity)
        gt_label_index=convert_to_scalar(gt_label_index)

        print("add similarity")
        print(similarity)
        print(name_id)
        print(gt_label_index)
        find_person=False

        for person in self.person_pred_list:
            if int(person.name_id)==int(name_id):
                find_person=True
                print("\033[0;31;40mfind person\033[0m")
                person.add_pred(similarity)
                if person.gt_label_index!=gt_label_index:

                    print(person.name_id)
                    print(gt_label_index)
                    print(person.gt_label_index)
                    print("\033[0;31;40mstorage error\033[0m")
                    exit(1)
                break

        if find_person is False:
            new_person=Person_pred(name_id,gt_label_index)
            new_person.add_pred(similarity)
            self.person_pred_list.append(new_person)



class Person_pred():
    def __init__(self,name_id,gt_label_index):
        super().__init__()
        self.name_id=name_id
        self.pred_list=[]
        self.gt_label_index=gt_label_index
        self.average_similarity=0
        self.count=0
        self.match_number=0
        print("\033[0;31;40mnew person\033[0m")
        print("name: "+str(self.name_id))
        print("gt label: "+str(gt_label_index))
    def add_pred(self, similarity):
        self.pred_list.append(similarity)
        self.average_similarity=np.mean(self.pred_list)
        self.count+=1
        if similarity>=0.5 and self.gt_label_index==1\
            or similarity<0.5 and self.gt_label_index==0:
            self.match_number+=1






@torch.no_grad()
def validate_personwise(val_loader,  model, config,logger,writer,full_text_tokens_set,val_data,epoch):
    model.eval()


    local_gt = []
    local_pred = []

    pred_storage=Pred_storage()

    with torch.no_grad():
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id_asd = get_asd_label_index_with_id(label_id)
            print("\033[0;31;40m label asd\033[0m")
            print(label_id_asd)
            name_id=get_name_id(label_id)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, 1)).cuda()

            for i in range(n):
                print("\033[0;31;40mn in val id\033[0m")
                print(n)
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id_asd = label_id_asd.cuda(non_blocking=True)
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

            for pred_idx in range(similarity.shape[0]):
                print(name_id[pred_idx])
                pred_storage.add_pred(name_id[pred_idx],similarity[pred_idx],label_id_asd[pred_idx])


        local_total_match=0
        local_total_count=0
        local_person_match=0
        local_person_count=0
        print("\033[0;31;40mperson number\033[0m")
        print(len(pred_storage.person_pred_list))


        for person in pred_storage.person_pred_list:
            print("\033[0;31;40mperson name_id\033[0m")
            print("person name"+str(person.name_id))
            local_pred.append(person.average_similarity)
            local_gt.append(person.gt_label_index)
            #print("\033[0;31;40mperson name_id\033[0m")
            print(person.average_similarity,person.gt_label_index)
            print(person.match_number,person.count)
            print(person.pred_list)
            local_total_count+=person.count
            local_total_match+=person.match_number

            local_person_count+=1
            if person.average_similarity>=0.5 and person.gt_label_index==1 \
                or person.average_similarity<0.5 and person.gt_label_index==0:
                local_person_match+=1



        print("\033[0;31;40mvideowise acc\033[0m")
        print(local_total_match,local_total_count)
        print("acc="+str(local_total_match/local_total_count*100.))

        print("\033[0;31;40mpersonwise acc\033[0m")
        print(local_person_match, local_person_count)
        print("acc=" + str(local_person_match / local_person_count * 100.))


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

        return avg_auc











