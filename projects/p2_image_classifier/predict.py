# -*- coding: utf-8 -*-
"""predict.py 

This python script reads in a pyTorch model checkpoint file and predict the flower based on
image given.

@author: Wong Songhan <songhan89@gmail.com>
@date: 1 Mar 2020


Example:
        To use a pre-trained vgg16 model to predict the top `k` number of classes
        for the input image `./flowers/test/17/image_03864.jpg `
        $ python predict.py --image-path ./flowers/test/17/image_03864.jpg 
                            --checkpt-path ./checkpoint/vgg16_0.001_1024_5.pth
                            --gpu
                            --top_k 3

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
        
    parser = argparse.ArgumentParser("Parser for model prediction.")

    parser.add_argument('--image-path', dest='image_path', action="store",
                        help="File path for image", default="./flowers/test/17/image_03864.jpg")
    parser.add_argument('--checkpt-path', dest='checkpt_path', action="store",
                        help="File path for model checkpoint .json",
                        default="checkpoint")
    parser.add_argument('--gpu', action='store_true',
                        help="Use GPU for inference")
    parser.add_argument('--top_k', dest='top_k', action="store",
                        help="top 'k' number of classes/probabilities", type=int, default=5)
    parser.add_argument('--category_names', dest='cat_to_name_json', action="store",
                        help="File path for category to name .json", default="cat_to_name.json")
    
    args = parser.parse_args()
    
    print (args)
    
    with open(args.cat_to_name_json, 'r') as f:
        cat_to_name = json.load(f)
    
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print (f'torch device:{device} for inference')
    
    model = flower_torch_util.load_checkpoint(args.checkpt_path, args.gpu)
    
    print (f'{model}')
        
    prob_tensor_list, class_tensor_list = flower_torch_util.predict(args.image_path, model, args.top_k, device)
    
    class_map = {val: key for key, val in model.class_to_idx.items()}
    class_map = {key: cat_to_name[val] for key, val in class_map.items()}
    class_list = [class_map[x] for x in class_tensor_list.cpu().numpy()]
    prob_list = prob_tensor_list.cpu().numpy()
    
    print ("The list of predictions is as followed (probability, class):")
           
    for prob, class_item in zip(prob_list, class_list):
        print (prob, class_item)
    
    
if __name__ == "__main__":
    main()