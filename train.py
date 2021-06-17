import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
import math
import argparse
import yaml
import torch.distributed as dist
from data.attr_dataset import create_dataloader

from model.model import buildModel
from utils.torch_utils import select_device
import torch.nn.functional as F

#TODO: remove cfg dependence(simplify code)
from data.carattr import CarAttr
from fastreid.config import get_cfg
from fastreid.data.transforms import build_transforms
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.config import CfgNode as CN
from data.attr_dataset import AttrDataset
from fastreid.data.build import build_reid_train_loader

logger = logging.getLogger(__name__)

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def add_attr_config(cfg):
    _C = cfg

    _C.MODEL.LOSSES.BCE = CN({"WEIGHT_ENABLED": True})
    _C.MODEL.LOSSES.BCE.SCALE = 1.

    _C.TEST.THRES = 0.5

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    args.config_file="config/carattr_res18.yml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def build_optimizer(hyp,model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay


    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    return optimizer

def build_lr_scheduler(hyp,optimizer,epochs):
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler

def build_train_dataloader(cfg,root):
    train_dataset = CarAttr(root)
    train_dataset.show_train()
    attr_dict = train_dataset.attr_dict
    train_items = list()
    train_items.extend(train_dataset.train)
    train_transforms = build_transforms(cfg, is_train=True)
    train_set = AttrDataset(train_items, train_transforms, attr_dict)
    data_loader = build_reid_train_loader(cfg, train_set=train_set)
    return data_loader

def preprocess_image(batched_inputs):
    """
    Normalize and batch the input images.
    """
    if isinstance(batched_inputs, dict):
        images = batched_inputs['images']
    elif isinstance(batched_inputs, torch.Tensor):
        images = batched_inputs
    else:
        raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255]
    pixel_std=[0.229*255, 0.224*255, 0.225*255]
    pixel_mean=torch.Tensor(pixel_mean).view(1, -1, 1, 1).to(images.device)
    pixel_std = torch.Tensor(pixel_std).view(1, -1, 1, 1).to(images.device)
    images.sub_(pixel_mean).div_(pixel_std)
    return images

def ratio2weight(targets, ratio):
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    weights[targets > 1] = 0.0
    return weights

def cross_entropy_sigmoid_loss(pred_class_logits, gt_classes, sample_weight=None):
    loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes, reduction='none')

    if sample_weight is not None:
        targets_mask = torch.where(gt_classes.detach() > 0.5,
                                   torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
        weight = ratio2weight(targets_mask, sample_weight)
        loss = loss * weight

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt
    return loss

def computelosses(outputs, gt_labels,sample_weights=None):
    r"""
    Compute loss from modeling's outputs, the loss function input arguments
    must be the same as the outputs of the model forwarding.
    """
    # model predictions
    cls_outputs = outputs["cls_outputs"]

    loss_dict = {}

    bce_scale=1.0

    loss_dict["loss_bce"] = cross_entropy_sigmoid_loss(
        cls_outputs,
        gt_labels,
        sample_weights,
    ) * bce_scale

    return loss_dict

def train(hyp,opt):
    root='../data'
    start_epoch = 0
    epochs = opt.epochs
    IMS_PER_BATCH=opt.batch_size

    save_dir='log'
    results_file = 'log/results.txt'

    #init device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #init dataset loader
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    #TODO: remove cfg dependence(simplify code),refactor dataset loader
    train_dataset=build_train_dataloader(cfg,root)
    ##########
    # img_size=[256,192]
    # workers=8
    # batch_size=opt.batch_size
    # train_dataset=create_dataloader(root, img_size,batch_size,rank=opt.global_rank,world_size=opt.world_size,workers=workers)
    #########

    feat_dim=512
    num_classes=train_dataset.dataset.num_classes
    #init model
    model=buildModel(feat_dim,num_classes).to(DEVICE)
    #init optimizer
    optimizer=build_optimizer(hyp,model)
    iters_per_epoch = len(train_dataset.dataset) // IMS_PER_BATCH
    _data_loader_iter = iter(train_dataset)
    #init scheduler
    scheduler=build_lr_scheduler(hyp, optimizer, epochs)


    save_model_path="runs/"
    debug_steps = 100
    validation_epochs = 5
    for epoch in range(start_epoch,epochs):
        model.train()
        running_loss=0.0
        mloss = torch.zeros(1, device=DEVICE)  # mean losses
        for i in range(iters_per_epoch):
        # for i, data in enumerate(train_dataset):
            data = next(_data_loader_iter)
            images = preprocess_image(data)
            pred = model(images)
            targets = data["targets"]
            loss_dict=computelosses(pred,targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()

            losses.backward()
            optimizer.step()


            running_loss += losses.item()
            if i and i % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                logger.info(
                    f"Epoch: {epoch}, Step: {i}, " +
                    f"Average Loss: {avg_loss:.4f}, "
                )
                running_loss = 0.0

            mloss = (mloss * i + losses.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 1) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss)
            if i and i % debug_steps == 0:
                logger.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
                logger.info(s)

        scheduler.step()



        if epoch % validation_epochs == 0 or epoch == epochs - 1:
            model_path = os.path.join(save_model_path, f"Epoch-{epoch}.pth")
            model_state_dict=model.state_dict()
            torch.save(model_state_dict, model_path)
            # model.save(model_path)
            logger.info(f"Saved model {model_path}")

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '\n')






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    train(hyp,opt)