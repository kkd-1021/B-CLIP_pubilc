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
from utils.extra_utils import get_name_id,label_length,get_asd_label_index
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
from utils.extra_utils import get_text_id
import clip
from utils.extra_utils import get_full_describe_text
def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader,  config,
                    mixup_fn,logger,writer,full_text_tokens_set,cls_optimizer,cls_lr_scheduler):

    model.train()
    optimizer.zero_grad()
    cls_optimizer.zero_grad()



    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()



    for idx, batch_data in enumerate(train_loader):


        images = batch_data["imgs"].cuda(non_blocking=True)

        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        print("\033[0;31;40mlabel id\033[0m")
        print(label_id)


        gt_label_index=get_text_id(label_id)#index
        gt_label_index=gt_label_index.to(label_id.device)
        print("\033[0;31;40mgt label\033[0m")
        print(gt_label_index)#one in 256

        label_id_asd_ind = get_asd_label_index(label_id)
        label_id_asd_ind = label_id_asd_ind.to(label_id.device)
        if full_text_tokens_set.shape[0] == 1:
            full_text_tokens_set = full_text_tokens_set.view(1, -1)
        images, gt_label_vec = mixup_fn(images, gt_label_index)

        output,cls_output = model(images, full_text_tokens_set,val=False)

        contrastive_loss = criterion(output, gt_label_vec)
        #classification
        print("\033[0;31;40mcls output\033[0m")
        print(cls_output)
        print("\033[0;31;40mlabel id asd ind\033[0m")
        print(label_id_asd_ind)
        cls_criterion=nn.BCEWithLogitsLoss(reduction='sum')
        cls_loss = cls_criterion(cls_output, label_id_asd_ind)
        print("\033[0;31;40mcls loss total\033[0m")
        print(cls_loss)
        print("\033[0;31;40mcontrastive loss total\033[0m")
        print(contrastive_loss)
        writer.add_scalar('cls loss', cls_loss, idx + epoch * len(train_loader))
        writer.add_scalar('contrastive loss', contrastive_loss, idx + epoch * len(train_loader))

        if config.TRAIN.ONLY_FINETUNE:
            print("only finetune")
            total_loss=contrastive_loss*0.+cls_loss
        else:
            total_loss=contrastive_loss+cls_loss*0.5

        print("\033[0;31;40mloss\033[0m")
        print(total_loss)

        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS



        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("\033[0;31;40mlr=\033[0m" + str(lr))
        writer.add_scalar('xclip_lr', optimizer.state_dict()['param_groups'][0]['lr'], idx + epoch * len(train_loader))

        if not config.TRAIN.ONLY_FINETUNE:
            print("\033[0;31;40mclip optimizing\033[0m" + str(lr))
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    #optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)

            else:
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            print("\033[0;31;40m only cls optimizing\033[0m" + str(lr))



        if (idx + 1) % (config.TRAIN.ACCUMULATION_STEPS/2) == 0:
            cls_optimizer.step()
            cls_optimizer.zero_grad()
            writer.add_scalar('cls_lr', cls_optimizer.param_groups[0]['lr'], idx + epoch * len(train_loader))



        if config.TRAIN.ACCUMULATION_STEPS > 1:
            print("\033[0;31;40macc step\033[0m" )
            print(config.TRAIN.ACCUMULATION_STEPS)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()



        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.15f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            writer.add_scalar('val', tot_loss_meter.val, idx + epoch * len(train_loader))
            writer.add_scalar('avg', tot_loss_meter.avg, idx + epoch * len(train_loader))


    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


