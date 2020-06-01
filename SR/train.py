import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import functools
from functools import partial
import numpy as np
import glob
import cv2
import os
import torch.nn as nn
import math
import argparse

from model.MPNCOV.MPNCOV import CovpoolLayer
from model.common import save_checkpoint, Hdf5Dataset, adjust_learning_rate
from model.RRDB import RRDBNet
from model.vgg_feature_extractor import VGGFeatureExtractor
from datetime import datetime

from torchsummary import summary

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--nEpochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--start_epoch", type=int, default=0, help='Starting Epoch')
parser.add_argument("--net", type=str, default="ours", help="RCAN, ESRGAN, SAN, RDN, EPSR, SRFAN, ours")
parser.add_argument("--lr_h5", type=str, default="../../net_data/blurred_final_64_32_grayscale.h5", help='path of LR h5 file')
parser.add_argument("--hr_h5", type=str, default="../../net_data/ground_truth_final_64_32_grayscale.h5", help='path of HR h5 file')
parser.add_argument("--ngpu", type=int, default=1, help='number of GPUs')
parser.add_argument("--batch_size", type=int, default='16', help='number of GPUs')
parser.add_argument("--resume", type=str, default="", help='restart training checkpoint path')
parser.add_argument('--extra_model_name', type=str, default='', help='addition name for path')


opt = parser.parse_args()

num_workers = 1
batch_size = opt.batch_size
initial_lr = 0.0001 #before 1e-4 25.05
train_set = Hdf5Dataset(lrname=opt.lr_h5, hrname=opt.hr_h5)

gpu_num = opt.ngpu

device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(gpu_num)
print(device)
print("===> Building model")
if (opt.net == 'ours'):
    model = RRDBNet()
    mse_factor = 0.5
    feature_factor = 0.05
    texture_factor = 0.05
    #vgg = VGGFeatureExtractor(device=device, feature_layer=[2, 7, 16, 25, 34], use_bn=False, use_input_norm=True)
    #vgg = nn.DataParallel(vgg,device_ids=[gpu_num])
    #vgg.to(device)
if (opt.net == 'RCAN'):
    model = RCAN()
if (opt.net == 'SRFBN'):
    model = SRFBN()
if (opt.net == 'SAN'):
    model = SAN()
elif (opt.net == 'RDN'):
    model = RDN()
elif (opt.net == 'EPSR'):
    model = EDSR()
    GANcriterion = GANLoss('ragan', 1.0, 0.0)
    l1_factor = 0
    mse_factor = .05
    feature_factor = 1
    gan_factor = 0.04
elif (opt.net == 'ESRGAN'):
    model = RRDBNet(nb=23)
    GANcriterion = GANLoss('ragan', 1.0, 0.0)
    l1_factor = 0.01
    mse_factor = 0
    feature_factor = 1.
    gan_factor = 0.005
if (opt.net == 'EPSR' or opt.net == 'ESRGAN'):
    gan = Discriminator_VGG_128()
    gan = nn.DataParallel(gan,device_ids=range(opt.ngpu))
    gan.to(device)
    vgg = VGGFeatureExtractor(device=device, feature_layer=34, use_bn=False, use_input_norm=True)
    vgg = nn.DataParallel(vgg,device_ids=range(opt.ngpu))
    vgg.to(device)
    optimizerD = optim.Adam(gan.parameters(), lr=initial_lr, weight_decay=1e-5)
    training_data_loader_gan = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

#model = nn.DataParallel(model, device_ids=range(opt.ngpu))
#model = nn.DataParallel(model, device_ids=range(opt.ngpu))
if (len(opt.resume) > 0):
    model.load_state_dict(torch.load(opt.resume)['model'].state_dict())

print('memory load before loading model '+str(torch.cuda.memory_allocated(device=device)))
model.to(device)
print('memory load after loading model '+str(torch.cuda.memory_allocated(device=device)))

#added - 2.05.2020 ------------
#summary(model, input_size=(1,64, 64))
#-----------------------
MSEcriterion = nn.MSELoss()
L1criterion = nn.L1Loss()
# Creating data indices for training and validation splits: added 17.05 ----------------------
validation_split = .1
shuffle_dataset = True
random_seed= 42

dataset_size = len(train_set)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_data_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                                sampler=valid_sampler)
#training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True) ---------------------


for epoch in range(opt.start_epoch, opt.nEpochs):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    lr = adjust_learning_rate(initial_lr, optimizer, epoch)
    #return to train mode
    model.train() 
    
    for iteration, batch in enumerate(training_data_loader, 1):
        #x_data, z_data = Variable(batch[0].float()).cuda(), Variable(batch[1].float()).cuda()
        x_data, z_data = Variable(batch[0].float()).to(device), Variable(batch[1].float()).to(device)
        output = model(z_data)
        
        
        
        if (opt.net == 'SRFBN'):
            loss = L1criterion(output[0], x_data) + L1criterion(output[1], x_data) + L1criterion(output[2], x_data) + L1criterion(output[3], x_data)
            mseloss = MSEcriterion(output[3], x_data)
        else:
            loss = MSEcriterion(output, x_data)
            mseloss = MSEcriterion(output, x_data)
        
        #if (opt.net == 'ours'):
        
         #   color_gt = torch.cat((x_data,x_data,x_data),1).cuda() #for grayscale
          #  color_output = torch.cat((output,output,output),1).cuda()
            
          # vgg_gt = vgg(color_gt)
           # vgg_output = vgg(color_output)
            #for i in range(5):
             #   loss = loss + 0.2*feature_factor*MSEcriterion(vgg_gt[i], vgg_output[i])
                #if (i==4):              --- removed 10.05 second order mse to make the training faster (pilar)
                    #loss += 0.2*feature_factor*MSEcriterion(CovpoolLayer(vgg_gt[i]),CovpoolLayer(vgg_output[i]))
        

            
        if iteration % 10000 == 0:
            print("{}===> Epoch[{}]({}/{}): MSELoss: {:.10f}".format(datetime.now().isoformat(), epoch, iteration, len(training_data_loader), mseloss.item()))
            #save_checkpoint(model, epoch, 'RRDB')
            
     #validation mode
    save_checkpoint('../../net_data/trained_srs/' + opt.net + opt.extra_model_name + '/', model, epoch, opt.net)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_data_loader:
            x_val, z_val = Variable(inputs.float()).to(device),Variable(labels.float()).to(device)
            x_est = model(z_val)
            batch_loss = MSEcriterion(x_est, x_val)
            val_loss += batch_loss.item()
                   
    print(f"{datetime.now().isoformat()} Epoch [{epoch+1}].. "
                  f"Test loss: {val_loss/len(validation_data_loader):.10f}.. ")

       
    

