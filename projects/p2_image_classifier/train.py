# -*- coding: utf-8 -*-
"""train.py 

This python script trains a deep nn model based on parameters provided by user
Model parameters are saved in the checkpoint file.

@author: Wong Songhan <songhan89@gmail.com>
@date: 1 Mar 2020


Example:
        To train a `alexnet` model with a learning rate of 0.005 and hidden units of 512,
        with gpu activated and 3 epochs, the following cmd can be used:
        $ python train.py --arch alexnet --learning_rate 0.005 --hidden_units 512 --epochs 3 --gpu

"""

import torch
import helper
import numpy as np
import argparse
import flower_torch_util
import os
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser("Parser for training parameters.")

    parser.add_argument('--data-dir', dest='data_dir', action="store",
                        help="Directory for data source", default="flowers")
    parser.add_argument('--checkpt-dir', dest='checkpt_dir', action="store",
                        help="Directory for checkpoint .json",
                        default="checkpoint")
    parser.add_argument('--arch', dest='arch', action="store",
                        help="Network architecture", default="vgg16")
    parser.add_argument('--learning_rate', dest='learning_rate', action="store",
                        help="Learning rate e.g 0.01", type=float,
                        default=0.001)
    parser.add_argument('--hidden_units', dest='hidden_units', action="store",
                        help="Number of hidden layers", type=int, default=1024)
    parser.add_argument('--epochs', dest='epochs', action="store",
                        help="Epochs", type=int, default=5)
    parser.add_argument('--gpu', action='store_true',
                        help="Use GPU for training")

    parser.parse_args()

    data_dir = os.path.join(os.getcwd(), args.data_dir)
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'train')
    checkpoint_file = ('%s_%s_%s_%s.pth' % (args.arch, args.learning_rate,
                                            args.hidden_units, args.epochs))
    checkpoint_save = os.path.join(os.getcwd(), args.checkpt_dir,
                                   checkpoint_file)
    
    print (args)
    
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print (f'torch device:{device}')

    try:
        print ('Load and transform data sources')
        trainloader, validloader, testloader, train_dataset, valid_dataset, \
        test_dataset = flower_torch_util \
            .load_transform(train_dir, valid_dir, test_dir)
    except e as Exception:
        print ('Data cannot be loaded properly!')
        
    if args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.in_features = 9216
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.in_features = 25088
        
    print (f'{model}')

    criterion = nn.NLLLoss()

    try:
        print ('Start model training')
        flower_torch_util.train_model(model, args.hidden_units,
                                      criterion, args.learning_rate, device,
                                      args.epochs,
                                      trainloader, validloader)
        print ('Compute test scores')
        flower_torch_util.test_model(model, testloader, device)
    except e as Exception:
        print (e)
        

    try:
        print (f'Saving model checkpoint {checkpoint_save}')
        flower_torch_util.save_checkpoint(model, args.arch, train_dataset.class_to_idx,
                                          args.hidden_units, args.epochs,
                                          checkpoint_save)
    except e as Exception:
        print (e)
        

if __name__ == "__main__":
    main()
