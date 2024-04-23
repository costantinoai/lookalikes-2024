#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:32:46 2024

@author: costantino_ai
"""

from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

vid, _, _ = read_video("test/assets/videos/v_SoccerJuggling_g23_c01.avi", output_format="TCHW")
vid = vid[:32]  # optionally shorten duration

# Step 1: Initialize model with the best available weights
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(vid).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
label = prediction.argmax().item()
score = prediction[label].item()
category_name = weights.meta["categories"][label]
print(f"{category_name}: {100 * score}%")