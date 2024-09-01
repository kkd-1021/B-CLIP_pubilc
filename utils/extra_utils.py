

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import pandas as pd
import scipy


label_length=8

NUM_CLASS=256
NAME_ID_RANGE=1000


def get_name_id(label_id):

    name_id=torch.zeros(label_id.shape).to(label_id.device)
    for idx in range(label_id.shape[0]):
        name_id[idx]=label_id[idx]%NAME_ID_RANGE

    return name_id




describe_text256=[
    ['is normal','is autism'],
    ['his body movements are flexible','his body movements are not flexible'],
    ['is good at completing cognitive function tasks','has difficulty in completing cognitive function tasks'],
    ['his hand movements are flexible','his hand movements are not flexible'],
    ['can understand what others say','can not understand what others say'],
    ['can say','can not say'],
    ['has a good interaction with an adult','has a bad interaction with an adult'],
    ['has no special interests','has strong special interests']
]


def get_text_id(label_id):

    gt_id_set=[]
    for idx in range(label_id.shape[0]):
        label_num = label_id[idx].cpu().numpy()

        label_num_str=str(label_num)
        reversed_string = label_num_str[::-1]
        gt_id=0
        num_ut = [1, 2, 4, 8, 16, 32, 64, 128]

        for char_idx,char in enumerate(reversed_string):
            gt_id+=num_ut[char_idx]*int(char)

        if gt_id>=NUM_CLASS:
            print("gt id error")
            exit(1)
        gt_id_set.append(gt_id)

    gt_id_set = torch.tensor(gt_id_set)
    return gt_id_set



def get_full_describe_text(clip):
    describe_set = []
    for i in range(0, NUM_CLASS):

            asd_label = i // ((2 ** 2) * (2 ** 5))
            mullen_label_list = []
            for idx in range(0, 5):
                mullen_label = i // ((2 ** 2) * (2 ** idx))
                mullen_label = mullen_label % 2
                mullen_label_list.append(mullen_label)

            css_sa_label = i // 2
            css_sa_label = css_sa_label % 2

            css_rbb_label = css_sa_label % 2

            if asd_label > 1:
                print("asd_label error")
                exit(1)

            describe = "a child "  + describe_text256[0][asd_label]

            for mullen_idx,mullen_label in enumerate(mullen_label_list):
                describe = describe + ' and ' + describe_text256[mullen_idx+1][mullen_label]

            describe = describe + ' and ' + describe_text256[6][css_sa_label]
            describe = describe + ' and ' + describe_text256[7][css_rbb_label]

            print(describe)

            text_aug = f"{{}}"
            describe_tokens = clip.tokenize(text_aug.format(describe), context_length=77)
            describe_set.append(describe_tokens[0])



    describe_set = torch.stack(describe_set)
    return describe_set




def get_asd_label_index(label_id):
    div=10**7
    list=[]
    for idx in range(label_id.shape[0]):
        label_id_asd=torch.zeros(1).to(label_id.device)
        if NUM_CLASS==864:
            if label_id[idx] / div >= 2:
                label_id_asd[0]=1#is asd
            else:
                label_id_asd[0]=0
        else:
            if label_id[idx] / div >= 1:
                label_id_asd[0]=1#is asd
            else:
                label_id_asd[0]=0

        list.append(label_id_asd)
    label_id_asd = torch.stack(list)
    label_id_asd = torch.tensor(label_id_asd, dtype=torch.float32)
    return label_id_asd


def get_asd_label_index_with_id(label_id):
    div=10**10
    list=[]
    for idx in range(label_id.shape[0]):
        label_id_asd=torch.zeros(1).to(label_id.device)
        if label_id[idx] / div >= 1:
            label_id_asd[0]=1#is asd
        else:
            label_id_asd[0]=0
        list.append(label_id_asd)
    label_id_asd = torch.stack(list)
    label_id_asd = torch.tensor(label_id_asd, dtype=torch.float32)
    return label_id_asd



def backward_hook(module, grad_in, grad_out):
    print("\033[0;31;40mgrad\033[0m")
    print(grad_out)
    print(grad_in)


def build_cls_optimizer_scheduler(model,config):
    model = model.module if hasattr(model, 'module') else model
    if not config.TRAIN.ONLY_FINETUNE:
        optimizer = torch.optim.Adam(model.classification_net.parameters(), lr=0.1, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

    else:
        optimizer = torch.optim.Adam(model.classification_net.parameters(), lr=0.005, eps=1e-08)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, last_epoch=-1)
    return optimizer,scheduler

