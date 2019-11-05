import argparse
import os, cv2
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

n_epochs = 100
data_dir = sys.argv[1]
batch_size= 32
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
img_size=250
channels = 3
interval = 100

cuda = True if torch.cuda.is_available() else False


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.299, 0.225]

ages = pickle.load(open("ages.pkl", "rb"))

class Discriminator():
    def model(self):
       num_classes = 1
       miodel_ft = models.inception_v3(pretrained=True)
       num_ftrs = model_ft.AuxLogits.fc.in_features
       model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
       num_ftrs = model_ft.fc.in_features
       model_ft.fc = nn.Linear(num_ftrs,num_classes)
       discriminator = model_ft.cuda()
       return discriminator
   
   
discriminator =  Discriminator().model()  #Discriminator().model()

if cuda:
    discriminator.cuda()

# Initialize weights

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.299, 0.225]

def datatransforms(crop_size):
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
       # transforms.Normalize( mean, std)
      ]),
      'val': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
       # transforms.Normalize(mean, std)
      ]),
      'test': transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
       # transforms.Normalize(mean, std)
      ]),

    }
    return data_transforms

data_transforms = datatransforms(299)
print(ImageFolderWithPaths(os.path.join(data_dir, 'train')))

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}

dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             num_workers=16)
              for x in ['train']}

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#  Training

for epoch in range(n_epochs):
    phase = 'train'
    for i, (imgs, labels, paths) in enumerate(dataloader[phase]):
        deg_imgs = []
        targets = []
        for idx, path in enumerate(paths):
            if path.split('/')[-1] in ages:
                deg_imgs.append(torch.tensor(imgs[idx]))
                targets.append(ages[path.split('/')[-1]])

        deg_imgs = torch.stack(deg_imgs)

        y = Variable(imgs.type(Tensor))  # original images are target.
        # Loss  Discrminator
        optimizer_D.zero_grad()
        d_org, _ = discriminator(y) # should be maximized.
        d_org = d_org.squeeze(1)
        targets = torch.FloatTensor(targets).cuda()
        loss = nn.MSELoss() #reduce=False)
        loss_d = loss(targets, d_org).sum() / targets.shape[0]

        loss_d.cuda()
        print("loss of discriminator:", loss_d.item())

        loss_d.backward() #retain_graph=True)

        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] ---------------------------------[D loss: %f] " % (epoch, n_epochs, i, len(dataloader), loss_d.item()))
        if i%1000 ==0 :
            save_checkpoint({'state_dict': discriminator.state_dict()}, False, str(epoch)+"_"+ str(i)+'discriminator_checkpoint.pth.tar')

