import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import time

from src_files.helper_functions.helper_functions_voc2007 import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss, CausalInvarianceLoss
from randaugment import RandAugment
# policy pool
DATA_AUGMENTATIONS = ['basic', 'cutout', 'randaugment']
OPTIMIZERS = ['Adam', 'SGD', 'AdamW']
LR_SCHEDULERS = ['OneCycleLR', 'StepLR', 'CosineAnnealingLR']
LOSS_FUNCTIONS = ['AsymmetricLoss', 'FocalLoss', 'BCELoss']



# mab implementation
class MAB:
    def __init__(self, options, epsilon=0.1):
        self.options = options
        self.epsilon = epsilon
        self.counts = [0] * len(options)
        self.values = [0.0] * len(options)

    def select_option(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.options) - 1)
        else:
            return np.argmax(self.values)

    def update(self, chosen_option, reward):
        self.counts[chosen_option] += 1
        n = self.counts[chosen_option]
        value = self.values[chosen_option]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_option] = new_value

class DynamicAugmentation:
    def __init__(self, image_size, cold_epochs=5):
        
        self.cold_epochs = cold_epochs
        self.basic_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            CutoutPIL(cutout_factor=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.advanced_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img, epoch):
        if epoch < self.cold_epochs:
            return self.basic_transform(img)
        return self.advanced_transform(img)



class DynamicCocoDetection(CocoDetection):
    def __init__(self, *args, image_size=448, cold_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_aug = DynamicAugmentation(image_size, cold_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = Image.fromarray(np.uint8(img))
        img = self.dynamic_aug(img, self.current_epoch)
        return img, target

class DiversityLoss(nn.Module):
    def __init__(self, num_classes, main_criterion):
        super().__init__()
        self.num_classes = num_classes
        self.main_criterion = main_criterion
        
    def forward(self, outputs, targets):
        main_loss = self.main_criterion(outputs, targets)
        pred_probs = torch.sigmoid(outputs)
        diversity_loss = -torch.mean(torch.std(pred_probs, dim=0))
        return main_loss + 0.1 * diversity_loss

def get_scheduler(optimizer, args, steps_per_epoch):
    if args.epochs <= args.cold_epochs:
        return lr_scheduler.ConstantLR(
            optimizer, factor=0.1, total_iters=args.epochs * steps_per_epoch
        )
    else:
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs - args.cold_epochs,
            pct_start=args.pct_start
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets) 
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def get_optimizer(optimizer_type, model_params, lr, weight_decay):
    if optimizer_type == 'Adam':
        return torch.optim.Adam(params=model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(params=model_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == 'AdamW':
        return torch.optim.AdamW(params=model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_loss_function(loss_type, num_classes):
    if loss_type == 'AsymmetricLoss':
        return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif loss_type == 'FocalLoss':
        return FocalLoss()
    elif loss_type == 'BCELoss':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
    parser.add_argument('--data', type=str, default='./datasets/VOC2007_COCO', help='Path to COCO dataset')
    parser.add_argument('--lr', default=5e-5, type=float, help='Initial learning rate')
    parser.add_argument('--max-lr', default=5e-5, type=float, help='Maximum learning rate')
    parser.add_argument('--model-size', default='tresnet_l', choices=['tresnet_s', 'tresnet_m', 'tresnet_l'], help='Backbone model size')
    parser.add_argument('--decoder-embedding', default=512, type=int, help='Decoder embedding size')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--num-classes', default=20, type=int, help='Number of classes')
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of workers')
    parser.add_argument('--image-size', default=448, type=int, help='Input image size')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Total epochs')
    parser.add_argument('--pct-start', default=0.3, type=float, help='LR schedule pct start')
    parser.add_argument('--cold-epochs', default=5, type=int, help='Cold start epochs')
    parser.add_argument('--weight-mAP', default=1.0, type=float, help='Reward weight for mAP')
    parser.add_argument('--weight-stability', default=0.5, type=float, help='Reward weight for stability')
    args = parser.parse_args()

    writer = SummaryWriter('runs/multilabel_experiment')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mab_data_aug = MAB(DATA_AUGMENTATIONS)
    mab_optimizer = MAB(OPTIMIZERS)
    mab_lr_scheduler = MAB(LR_SCHEDULERS)
    mab_loss_fn = MAB(LOSS_FUNCTIONS)

    model = create_model(args).to(device)
    ema = ModelEma(model, 0.9997)

    train_dataset = DynamicCocoDetection(
        os.path.join(args.data, 'train2014'),
        os.path.join(args.data, 'annotations/instances_train2014.json'),
        image_size=args.image_size,
        cold_epochs=args.cold_epochs
    )
    val_dataset = CocoDetection(
        os.path.join(args.data, 'val2014'),
        os.path.join(args.data, 'annotations/instances_val2014.json'),
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    selected_opt = mab_optimizer.select_option()
    selected_loss = mab_loss_fn.select_option()
    optimizer = get_optimizer(OPTIMIZERS[selected_opt], add_weight_decay(model, args.weight_decay), args.lr, args.weight_decay)
    criterion = get_loss_function(LOSS_FUNCTIONS[selected_loss], args.num_classes)
    scheduler = get_scheduler(optimizer, args, len(train_loader))

    scaler = GradScaler()
    highest_mAP = 0
    loss_history = []

    start_time = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loader.dataset.set_epoch(epoch)
        epoch_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.max(dim=1)[0].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            loss_history.append(loss.item())

            if i % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader)+i)
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print("Elapsed Time: ", time.time() - start_time)
        current_mAP = validate_multi(val_loader, model, ema.module)
        recent_losses = loss_history[-len(train_loader):]
        loss_var = np.var(recent_losses)
        reward = args.weight_mAP * current_mAP + args.weight_stability * (1 / (1 + loss_var))

        mab_data_aug.update(selected_opt, reward)
        mab_optimizer.update(selected_opt, reward)
        mab_lr_scheduler.update(selected_opt, reward)
        mab_loss_fn.update(selected_opt, reward)

        new_opt = mab_optimizer.select_option()
        new_loss = mab_loss_fn.select_option()

        if new_opt != selected_opt:
            optimizer = get_optimizer(OPTIMIZERS[new_opt], add_weight_decay(model, args.weight_decay), args.lr, args.weight_decay)
            selected_opt = new_opt
        if new_loss != selected_loss:
            criterion = get_loss_function(LOSS_FUNCTIONS[new_loss], args.num_classes)
            selected_loss = new_loss

        if current_mAP > highest_mAP:
            highest_mAP = current_mAP
            torch.save(ema.module.state_dict(), 'models/model-highest.ckpt')

        print(f'Epoch {epoch+1}: mAP {current_mAP:.2f}, Best {highest_mAP:.2f}')

    writer.close()

def validate_multi(val_loader, model, ema_model):
    model.eval()
    ema_model.eval()
    sigmoid = nn.Sigmoid()
    preds_ema, targets_ema = [], []
    preds, targets = [], []

    with torch.no_grad():
        for inputs, target in val_loader:
            outputs_ema = sigmoid(ema_model(inputs.to(next(model.parameters()).device)))
            preds_ema.append(outputs_ema.cpu())
            targets_ema.append(target.max(dim=1)[0].cpu())

            outputs = sigmoid(model(inputs.to(next(model.parameters()).device)))
            preds.append(outputs.cpu())
            targets.append(target.max(dim=1)[0].cpu())

    map_ema = mAP(torch.cat(targets_ema).numpy(), torch.cat(preds_ema).numpy())
    map_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    return map_ema if map_ema > map_regular else map_regular

if __name__ == '__main__':
    main()