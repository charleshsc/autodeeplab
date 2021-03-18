import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from auto_deeplab import AutoDeeplab
from config_utils.search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict

args = obtain_search_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

if args.checkname is None:
    args.checkname = 'deeplab-'+str(args.backbone)

criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
model = AutoDeeplab (30, 12, criterion, args.filter_multiplier,
                             args.block_multiplier, args.step)

state_dict = model.state_dict()
saver = Saver(args)
saver.save_experiment_config()
optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
saver.save_checkpoint({
    'epoch': 0,
    'state_dict': state_dict,
    'optimizer': optimizer,
    'best_pred': 0.0
},False)