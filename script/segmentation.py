
import time, os, random
from datetime import datetime
import tqdm
from glob import glob
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchsummary import summary
import urllib.request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('torch ver.', torch.__version__)


# Global parameters
IMG_DIR = '../input/original/' # Original image dir
MASK_DIR = '../input/mask/' # Mask image dir
PROXY_PIN = '../PIN.txt' # [userID]:[passward]@[proxy server adrress]:[port number]
N_CLASSES = 2 # Number of classes including background, i.e. N_CLASSES=2 for binary segmentation)
BATCH_SIZE = 1
SEED = 19


# Random seed
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Proxy setting (to download encorder weigths of pretrained model) 
def set_proxy(pin=PROXY_PIN):
    with open(pin, 'r') as f:
        proxy_pass = f.read()
        proxies = {'http': 'http://' + proxy_pass, 'https': 'http://' + proxy_pass}
    proxy = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(proxy)
    urllib.request.install_opener(opener)


def get_img_names(img_dir):
    img_names = []
    for img_path in glob(img_dir + '*.jpg'):
        filename = img_path.split('\\')[-1].split('.')[0]
        img_names.append(filename)
    return img_names


# Metrics
# # Accuracy
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


# # IoU averaged over classes 
def meanIoU(pred_mask, mask, smooth=1e-10, n_classes=N_CLASSES):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        iou_per_class = []
        for c in range(n_classes):
            pred_label = (pred_mask==c)
            true_label = (mask==c)

            if true_label.long().sum().item() == 0: # no exists label
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(pred_label, true_label).sum().float().item()
                union = torch.logical_or(pred_label, true_label).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        
        return np.nanmean(iou_per_class)


# # Dice coeffient average over classes 
def meanDice(pred_mask, mask, smooth=1e-10, n_classes=N_CLASSES):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        dice_per_class = []
        for c in range(n_classes):
            pred_label = (pred_mask==c)
            true_label = (mask==c)

            if true_label.long().sum().item() == 0: # no exists label
                dice_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(pred_label, true_label).sum().float().item()
                left = torch.sum(pred_label)
                right = torch.sum(true_label)
                dice = (2. * intersect + smooth) / (left + right + smooth)
                dice_per_class.append(dice)

        return np.nanmean(dice_per_class)


# Dataset
class ImgMaskDataset(Dataset):

    def __init__(self, img_dir, mask_dir, train_or_valid, 
                transform=None, patch=False,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train_or_valid = train_or_valid
        self.mean = mean
        self.std = std
        self.transform = transform
        self.patches = patch
    
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img_names = get_img_names(self.img_dir)
        X_train, X_valid = train_test_split(img_names, test_size=0.2, random_state=SEED)
        
        if self.train_or_valid == 'train':
            self.X = X_train
        elif self.train_or_valid == 'valid':
            self.X = X_valid
        else:
            raise Exception('arg of "train_or_valid" shall be assigned as "train" or "valid".' )

        img = cv2.imread(self.img_dir + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir + self.X[idx] + '_mask.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
        return img, mask

    def tiles(self, img, mask):
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768)
        img_patches = img_patches.permute(1,0,2,3)
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        return img_patches, mask_patches


# Dataloader
def create_data_loader(img_dir, mask_dir):
                
        # Transformer
        t_train = A.Compose([
            A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0, 0.5),(0, 0.5)),
            A.GaussNoise()
            ])

        t_valid = A.Compose([
            A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.GridDistortion(p=0.2)
            ])

        train_set = ImgMaskDataset(img_dir, mask_dir, 'train', t_train, patch=False)
        valid_set = ImgMaskDataset(img_dir, mask_dir, 'valid', t_valid, patch=False)

        # Dataloader
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, BATCH_SIZE, shuffle=True)

        return train_loader, valid_loader


# Trainer
class Train():

    def __init__(self, img_dir, mask_dir):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    # Learning rate
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # Train
    def train(self, epochs, model, criterion, optimizer, scheduler, patch=False):
        
        # Pretrained base model
        model = smp.Unet('efficientnet-b2', encoder_weights='imagenet',
                          classes=N_CLASSES, activation=None, encoder_depth=5,
                          decoder_channels=[256, 128, 64, 32, 16])
        model.to(device)

        # Data loader
        train_loader, valid_loader = create_data_loader()

        # Hyper parameters
        max_lr = 1e-3
        epochs = 20
        weight_decay = 1e-4
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=max_lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr, 
                                                        epochs=epochs,
                                                        steps_per_epoch=len(train_loader))
        run_date = str(datetime.now().strftime("%Y%m%d-%H%M%S"))                 
        model_path = f'../model/model_{run_date}.pt' # saved model path
        
        # Initiation
        torch.cuda.empty_cache() # memory freeing
        train_losses, val_losses  = [], []
        train_iou, val_iou = [], []
        train_dice, val_dice = [], []
        train_acc, val_acc = [], []
        lrs = []
        min_loss = np.inf
        decrease = 0
        not_improve = 0

        # Train
        train_start_time = time.time()
        # # Loop per epoch
        for e in range(epochs):
            model.train()
            epoch_start = time.time()
            train_loss = 0
            iou_score = 0
            dice_score = 0
            accuracy = 0
            # Loop per batch
            with tqdm(train_loader) as pbar:
                for i, data in enumerate(pbar):
                    pbar.set_description('[Epoch {:d} train]'.format(e + 1))
                    # Load input
                    image_tiles, mask_tiles = data
                    if patch:
                        b, n_tiles, c, h, w = image_tiles.size()
                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    # Forward
                    output = model(image)
                    loss = criterion(output, mask)
                    accuracy += pixel_accuracy(output, mask) 
                    iou_score += meanIoU(output, mask)
                    dice_score += meanDice(output, mask)
                    # Backward
                    loss.backward() 
                    optimizer.step() # update weigth
                    optimizer.zero_grad() # reset gradient
                    lrs.append(self.get_lr(optimizer))
                    scheduler.step()
                    train_loss += loss.item()
                    pbar.set_postfix(OrderedDict(dice=dice_score/(i+1)))
            
            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            val_iou_score = 0
            val_dice_score = 0
            with torch.no_grad():
                with tqdm(valid_loader) as pbar:
                    for i, data in enumerate(pbar):
                        pbar.set_description('[Epoch {:d} valid]'.format(e + 1))
                        # Load input
                        image_tiles, mask_tiles = data
                        if patch:
                            b, n_tiles, c, h, w = image_tiles.size()
                            image_tiles = image_tiles.view(-1, c, h, w)
                            mask_tiles = mask_tiles.view(-1, h, w)
                        image = image_tiles.to(device)
                        mask = mask_tiles.to(device)
                        # Forward
                        output = model(image)
                        loss = criterion(output, mask)
                        val_loss += loss.item()
                        val_iou_score += meanIoU(output, mask)
                        val_dice_score += meanDice(output, mask)
                        val_accuracy += pixel_accuracy(output, mask)
                        pbar.set_postfix(OrderedDict(dice=val_dice_score/(i+1)))
            
            # Metrics averaged in batch
            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(valid_loader))
            
            # Save model if min_loss is updated
            if min_loss > (val_loss / len(valid_loader)):
                # print('Loss decreasing...{:.3f} >> {:.3f}'.format(min_loss, (val_loss/len(val_loader))))
                min_loss = val_loss / len(valid_loader)
                best_dice = val_dice_score / len(valid_loader)
                decrease += 1
                not_improve = 0
                if decrease >= 3:
                    # print('Saving model...')
                    torch.save(model, model_path)
                    
            # Early stopping if loss not updated 3 times in succession
            else:
                not_improve += 1
                print(f'Loss Not decrease for {not_improve} time')
                if not_improve == 3:
                    print('Stop training since loss is not decreased for 3 times in succession')
                    break
                
            # Score
            train_iou.append(iou_score / len(train_loader))
            val_iou.append(val_iou_score / len(valid_loader))
            train_dice.append(dice_score / len(train_loader))
            val_dice.append(val_dice_score / len(valid_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(val_accuracy / len(valid_loader))
            print('Epoch:{}/{} |'.format(e+1, epochs),
            'Train Loss: {:.3f} |'.format(train_loss/len(train_loader)),
            'Val Loss: {:.3f} |'.format(val_loss/len(valid_loader)),
            'Train Dice: {:.3f} |'.format(dice_score/len(train_loader)),
            'Val Dice: {:.3f} |'.format(val_dice_score/len(valid_loader)),
            'Train Acc: {:.3f} |'.format(accuracy/len(train_loader)),
            'Val Acc: {:.3f} |'.format(val_accuracy/len(valid_loader)),
    #        'Train mIoU: {:.3f} |'.format(iou_score/len(train_loader)),
    #        'Val mIoU: {:.3f} |'.format(val_iou_score/len(val_loader)),
            'Time: {:.2f} min.'.format((time.time()-epoch_start)/60))
        
        self.history = {'train_loss': train_losses, 'val_loss': val_losses,
                'train_miou': train_iou, 'val_miou': val_iou,
                'train_mdice': train_dice, 'val_mdice': val_dice,
                'train_acc' : train_acc, 'val_acc': val_acc,
                'lrs': lrs}
        print('Total time: {:.2f} min.' .format((time.time() - train_start_time)/60))
        best_model_path = model_path.split('.pt')[0] + '_dice-{:.3f}'.format(best_dice) + '.pt'
        os.rename(model_path, best_model_path)
        if decrease<3:
            print('Not model saved since training was not proceeding.')
        else:
            print(f'Model saved in {best_model_path}')

        return self.history, best_model_path

    def plot_loss(self):
        plt.plot(self.history['train_loss'], label='train', marker='o')
        plt.plot(self.history['val_loss'], label='val', marker='o')
        plt.title('Loss per epoch'); plt.ylabel('loss');
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()

    def plot_iou(self):
        plt.plot(self.history['train_miou'], label='train_mIoU', marker='*')
        plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
        plt.title('Score per epoch'); plt.ylabel('mean IoU')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()

    def plot_dice(self):
        plt.plot(self.history['train_mdice'], label='train_mdice', marker='*')
        plt.plot(self.history['val_mdice'], label='val_mdice',  marker='*')
        plt.title('Score per epoch'); plt.ylabel('mean dice')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()

    def plot_acc(self):
        plt.plot(self.history['train_acc'], label='train_accuracy', marker='*')
        plt.plot(self.history['val_acc'], label='val_accuracy',  marker='*')
        plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()



# Main
if __name__ == '__main__':

    # set_proxy()
    seed_everything()

    trainer = Train(IMG_DIR, MASK_DIR)
    history, model_path = trainer.train(epochs=5)
    trainer.plot_acc()
    trainer.plot_iou()
    





