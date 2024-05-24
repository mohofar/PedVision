import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
from torchvision import transforms
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

def prepare_data_and_model_classifier(rounds, cls_num, model_name, img_name):


    # Path to the directory containing the .npy files
    directory_path = 'test_data\predicted/'

    # Get a list of all final_image files
    test_files = directory_path + 'VFM/for_cls_net_'+img_name+'.npy'

    # def load_data(file):
    #     image = np.load(file)
    #     return image


    # Load and concatenate data for training and validation sets
    test_images = np.load(test_files)


    # Convert data to PyTorch tensors
    test_images = torch.tensor(test_images, dtype=torch.float32)

    test_images = TensorDataset(test_images)
    test_loader = DataLoader(test_images, batch_size=32, shuffle=True)

    if(model_name == 'MobileNet'):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, cls_num)
    elif(model_name == 'EffiB1'):
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, cls_num)
    elif(model_name == 'EffiB5'):
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, cls_num)
    else:
        print('Specify the model name: MobileNet, EffiB1, or EffiB5')

    model.load_state_dict(torch.load('PedVisionCode\saved_models\CLS_model_R'+str(rounds)+'.pth'))

    return test_loader, model

def main(rounds, cls_num, model_name, img_name, num_classes):
    
    test_loader, model = prepare_data_and_model_classifier(rounds, cls_num, model_name, img_name)

    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = torch.stack(inputs)
            outputs = model(inputs[0])
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())

    
    with open('test_data\predicted\VFM\org_mask_'+img_name+'.pkl', 'rb') as f:
        mask_np = pickle.load(f)
    image_final = np.zeros((mask_np[0]['segmentation'].shape[0], mask_np[0]['segmentation'].shape[1], num_classes), dtype=np.bool_)
    for i in range(len(mask_np)):
      image_final[:,:,all_preds[i]] = image_final[:,:,all_preds[i]] + mask_np[i]['segmentation'].astype(np.bool_)

    plt.figure(figsize=(10,10))
    for i in range(num_classes):
        plt.subplot(1,num_classes,i+1)
        plt.imshow(image_final[:,:,i])
        plt.axis('off')
        plt.title('Class '+str(i))
    plt.show()

                    
    
# if __name__ == "__main__":
#     main(rounds)