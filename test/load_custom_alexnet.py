#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:35:16 2024

@author: costantino_ai
"""

# Standard imports
import os
import torch
import torchvision
import torch.nn as nn
import copy 

# Local imports
from modules import logging
from modules.net_funcs.net_utils import print_model_summary, env_check
from modules.helper_funcs.utils import (
    OutputLogger,
    create_output_directory,
    create_run_id,
    print_dict,
    save_script_to_file,
    set_random_seeds
)

class AlexNetClass(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.activations = []
        self.gradients = []
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, features_layer=None, activations=False, gradients=False):
        self.activations = []
        if activations:
            num_features = len(self.features)
            num_classifier = len(self.classifier)
            for i in range(num_features):
                x = self.features[i](x)
                self.activations.append(copy.deepcopy(x))
                if gradients:
                    x.register_hook(self.activations_hook)
            x = x.view(x.size(0), -1)
            for i in range(num_classifier):
                x = self.classifier[i](x)
                self.activations.append(copy.deepcopy(x))
                if gradients:
                    x.register_hook(self.activations_hook)
            
        else:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 13 * 13)
            x = self.classifier(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients.append(grad)
    
    def get_activations_gradient(self):
        return self.gradients


def AlexNet(num_classes):
    model = AlexNetClass(num_classes)
    return model

########### PARAMETERS ###########

# Define a dictionary to store all parameters for the functions used in the script
params = {
    # General parameters
    "save_logs": False,
    "run_id": create_run_id() + '_extract-net-activations',
    "seed": 42
    
    # Functions-specific parameters
    # TODO: add parameters for each function in the main code
}

########### MAIN CODE ###########
set_random_seeds(params["seed"])

# Make output folder and save run files if log is True
out_dir = os.path.join("./results", params["run_id"])
out_text_file = os.path.join(out_dir, "output_log.txt")

# Save script and make output folder
if params["save_logs"]:
    
    # Make output folder
    create_output_directory(out_dir)
    
    # Save script to file
    save_script_to_file(out_dir)

    logging.info("Output folder created and script file saved")

with OutputLogger(params["save_logs"], out_text_file):
    # Check environment and print info
    env_check()

    # Printing run info and device information
    print_dict(params)
    
    logging.info("Started processing.")
    
    num_classes = 1714
    model = AlexNet(num_classes)
    model_dp = nn.DataParallel(model)
    print_model_summary(model_dp)
    
    ckp_path = "/data/modelling/models/alexnet_faces_epoch_100.pth.tar"
    ckp = torch.load(ckp_path)
    state_dict = ckp["state_dict"]
    print(state_dict.keys())
    
    model_dp.load_state_dict(state_dict)

    logging.info("Completed processing.")

    if params["save_logs"]:
        # Save data to file        
        pass # TODO: ADD YOUR SAVING CODE HERE                    
