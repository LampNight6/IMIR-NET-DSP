from __future__ import print_function
import argparse
import os
import shutil
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
from utils import *
from utils_data import get_DataLoader
#from light_cnn_v4 import LightCNN_V4
import torchvision as tv
from mynetwork import MyResNetRGBD
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', type=int, default=40,help="the frequency of write to logtxt" )
parser.add_argument('--model', default='rgbonly', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--data_root', default='C:/project/data/nutrition5k', type=str, metavar='PATH',
#                     help='path to root path of images (default: none)')
parser.add_argument('--data_root', default='/home/user/nfdProject/food', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--run_name', type=str, default="",
                    help="Experiment name; if empty, uses a timestamp (prevents overwriting).")
parser.add_argument('--exp_root', type=str, default="",
                    help="Optional root dir for logs/saved (default: current project dirs).")
parser.add_argument(
    "--ingredients_mode",
    type=str,
    default="clip512",
    choices=["clip512", "binary255"],
    help="Ingredients supervision format: clip512 (regress CLIP text features) or binary255 (0/1 vector).",
)
parser.add_argument(
    "--ingredients_vocab",
    type=str,
    default="",
    help="(binary255 only) JSON vocab: list[255] of names or {name: idx}. Required if ingredient items are strings.",
)
# parser.add_argument('--accum_steps', type=int, default=5,
#                     help="Gradient accumulation steps. Effective batch size = batch_size * accum_steps.")
parser.add_argument(
    "--accum_steps",
    type=int,
    default=5,
    help="Gradient accumulation steps. Effective batch size = batch_size * accum_steps.",
)

args = parser.parse_args()

if not args.run_name:
    args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

_base_dir = args.exp_root if args.exp_root else "."
_logs_root = os.path.join(_base_dir, "logs")
_saved_root = os.path.join(_base_dir, "saved")

log_dir = os.path.join(_logs_root, f"checkpoint_{args.model}_{args.run_name}")
log_file_path = os.path.join(log_dir, "train_log.txt")
check_dirs(log_dir)
logtxt(log_file_path, str(vars(args)))


def build_checkpoint_meta():
    pretrain_rel = "food2k_resnet101_0.0001.pth"
    pretrain_path = os.path.abspath(pretrain_rel) if os.path.isfile(pretrain_rel) else ""
    return {
        "run_name": args.run_name,
        "model": args.model,
        "pretrained_backbone": "food2k_resnet101" if pretrain_path else "",
        "pretrained_path": pretrain_path,
        "args": vars(args),
    }


_CKPT_META = build_checkpoint_meta()



def main():
    # validation/segment settings
    val_every = 5
    segment_size = 50  # 6 segments for 300 epochs
    save_dir = os.path.join(_saved_root, f"regression_{args.model}_{args.run_name}")
    check_dirs(save_dir)

    #global args
    #model = LightCNN_V4()
    # model = MyResNet(Bottleneck, [3, 4, 6, 3])  # 这里具体的参数参考库中源代码
    # model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
    ingredients_dim = 255 if args.ingredients_mode == "binary255" else 512
    model = MyResNetRGBD(ingredients_dim=ingredients_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(model)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # optionally resume from a checkpoint
    start_epoch = args.start_epoch
    resume_path = args.resume if args.resume else get_resume_path(save_dir)
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint.get('epoch', checkpoint.get('epoch_idx', 0))
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    cudnn.benchmark = True

    #load image
    train_loader, test_loader = get_DataLoader(args)

    # define loss function and optimizer
    criterion = nn.L1Loss()

    criterion = criterion.to(device)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            epoch,
        )

        do_validate = ((epoch + 1) % val_every == 0) or (epoch + 1 == args.epochs)
        if do_validate:
            validate(test_loader, model, criterion, device, epoch, save_dir, optimizer)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            "meta": _CKPT_META,
        }, os.path.join(save_dir, "ckpt_last.pth"))
        if (epoch + 1) % segment_size == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                "meta": _CKPT_META,
            }, os.path.join(save_dir, f"ckpt_segment_{epoch + 1}.pth"))

def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(train_loader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    accum_steps = max(int(getattr(args, "accum_steps", 1)), 1)
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, x in enumerate(epoch_iterator):  # (inputs, targets,ingredient)
        '''Portion Independent Model'''
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)
        inputs_ingredients = x[8].to(device)

        outputs = model(inputs, inputs_rgbd, inputs_ingredients)
        total_calories_loss = total_calories.shape[0] * criterion(outputs[0], total_calories) / total_calories.sum().item()
        total_mass_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
        total_fat_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
        total_carb_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
        total_protein_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()
        loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss

        (loss / accum_steps).backward()
        do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        calories_loss += total_calories_loss.item()
        mass_loss += total_mass_loss.item()
        fat_loss += total_fat_loss.item()
        carb_loss += total_carb_loss.item()
        protein_loss += total_protein_loss.item()

        if (batch_idx+1) % args.print_freq == 0 or batch_idx+1 == len(train_loader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \n'.format(
                    epoch, batch_idx+1, len(train_loader),
                    train_loss/(batch_idx+1),
                    calories_loss/(batch_idx+1),
                    mass_loss/(batch_idx+1),
                    fat_loss/(batch_idx+1),
                    carb_loss/(batch_idx+1),
                    protein_loss/(batch_idx+1)))

best_loss = 10000
def validate(test_loader, model, criterion, device, epoch, save_dir, optimizer=None):
    # switch to evaluate mode
    global best_loss
    model.eval()
    test_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator):  # testloader
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()
            inputs_rgbd = x[7].to(device)
            inputs_ingredients = x[8].to(device)

            outputs = model(inputs, inputs_rgbd, inputs_ingredients)

            # loss
            calories_total_loss = total_calories.shape[0] * criterion(outputs[0], total_calories) / total_calories.sum().item()
            mass_total_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
            fat_total_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
            carb_total_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
            protein_total_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()

            loss = calories_total_loss + mass_total_loss + fat_total_loss + carb_total_loss + protein_total_loss

            test_loss += loss.item()
            calories_loss += calories_total_loss.item()
            mass_loss += mass_total_loss.item()
            fat_loss += fat_total_loss.item()
            carb_loss += carb_total_loss.item()
            protein_loss += protein_total_loss.item()

        epoch_iterator.set_description(
            "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f" % (
            epoch, test_loss / (batch_idx + 1), calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),
            fat_loss / (batch_idx + 1), carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1)))

        logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                              'Loss: {:2.5f} \t'
                              'calorieloss: {:2.5f} \t'
                              'massloss: {:2.5f} \t'
                              'fatloss: {:2.5f} \t'
                              'carbloss: {:2.5f} \t'
                              'proteinloss: {:2.5f} \n'.format(
            epoch, batch_idx + 1, len(test_loader),
                   test_loss / len(test_loader),
                   calories_loss / len(test_loader),
                   mass_loss / len(test_loader),
                   fat_loss / len(test_loader),
                   carb_loss / len(test_loader),
                   protein_loss / len(test_loader)))
        # Save checkpoint.
        # pdb.set_trace()
    if best_loss > test_loss:
        best_loss = test_loss
        print('Saving..')
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            "meta": _CKPT_META,
        }
        save_checkpoint(state, os.path.join(save_dir, "ckpt_best.pth"))

def adjust_learning_rate(optimizer, epoch):
    scale = 0.5
    step  = 45
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

def save_checkpoint(state, filename):
    torch.save(state, filename)

def get_resume_path(save_dir):
    ckpt_last = os.path.join(save_dir, "ckpt_last.pth")
    if os.path.isfile(ckpt_last):
        return ckpt_last
    if not os.path.isdir(save_dir):
        return None
    segments = []
    for fname in os.listdir(save_dir):
        if fname.startswith("ckpt_segment_") and fname.endswith(".pth"):
            try:
                epoch_num = int(fname[len("ckpt_segment_"):-4])
            except ValueError:
                continue
            segments.append((epoch_num, os.path.join(save_dir, fname)))
    if not segments:
        return None
    segments.sort(key=lambda x: x[0], reverse=True)
    return segments[0][1]

if __name__ == '__main__':
    main()
