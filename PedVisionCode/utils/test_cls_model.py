import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os

def load_model_and_predict(mask, model_name, model_path, cls_num):
    mask = torch.tensor(mask, dtype=torch.float32)

    # Initialize model based on model_name
    if model_name == 'MobileNet':
        model_cls = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model_cls.classifier[1] = nn.Linear(model_cls.last_channel, cls_num)
    elif model_name == 'EffiB1':
        model_cls = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        model_cls.classifier[-1] = nn.Linear(model_cls.classifier[-1].in_features, cls_num)
    elif model_name == 'EffiB5':
        model_cls = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        model_cls.classifier[-1] = nn.Linear(model_cls.classifier[-1].in_features, cls_num)
    else:
        raise ValueError("Specify model_name as 'MobileNet', 'EffiB1', or 'EffiB5'")

    model_cls.load_state_dict(torch.load(model_path))
    model_cls.eval()  # Set model to evaluation mode

    # Predict
    with torch.no_grad():
        outputs = model_cls(mask)
        _, predicted = torch.max(outputs, 1)

    return predicted


def main(rounds, cls_num, model_name, img_name, num_classes):
    # File paths
    directory_path = 'PedVisionCode/test_data/predicted/'
    test_file_path = f'{directory_path}VFM/for_cls_net_{img_name}.npy'
    pkl_file_path = f'{directory_path}VFM/org_mask_{img_name}.pkl'

    # Load image and mask data
    test_images = np.load(test_file_path)
    with open(pkl_file_path, 'rb') as f:
        masks = pickle.load(f)

    # Predict class for each mask
    cls_model_path = f'PedVisionCode/saved_models/CLS_model_R{rounds}.pth'
    prediction = load_model_and_predict(test_images, model_name, cls_model_path, cls_num)

    # Display each class's mask
    for bone in range(cls_num):
        overall_mask = np.zeros(masks[0]['segmentation'].shape)
        for i, pred in enumerate(prediction):
            if pred.item() == bone:
                overall_mask += masks[i]['segmentation']

        plt.subplot(1, cls_num, bone + 1)
        plt.imshow(overall_mask.astype(np.bool_), alpha=0.7)
        plt.title(f'Class {bone}')

    plt.show()
