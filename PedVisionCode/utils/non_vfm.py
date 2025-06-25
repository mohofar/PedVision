
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        for img in os.listdir(image_dir):
            if img.endswith('.png') or img.endswith('.jpg'):
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # Read the image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Convert the image from NumPy array to a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.images[idx])

class CustomTransformTest:
    def __init__(self):
        # Separate resize transforms for image and mask
        self.resize_image = transforms.Resize((1024, 1024))
        # Convert to grayscale
        self.to_grayscale = transforms.Grayscale()

    def __call__(self, image):
        # Resize image and mask
        image = self.resize_image(image)
        # Convert to tensor after all PIL image transformations
        image = TF.to_tensor(image)
        # Convert to grayscale
        image = self.to_grayscale(image)

        return image
   
def test_model(model, test_loader, device):
    model.eval()
    outputs = []
    names = []
    with torch.no_grad():
        for inputs, name in tqdm(test_loader):
            inputs = inputs.to(device)
            output = model(inputs)

            # Convert the output to a NumPy array and append to the list
            outputs.append(output.squeeze().cpu().numpy())
            names.append(name[0])
    return outputs, names

def model_selection(model_name):
    if  model_name == 'deeplab34':
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34", # choose encoder, e.g., resnet34, mobilenet_v2, etc.
            encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1, # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4, # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )

        print(model)

        # load the best model
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/DeepLabV3Plus_resnet34_best_model.pth'))

    elif model_name == 'deeplab101':
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1, # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4, # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Deeplapv3p_res101_best_model.pth'))

    elif model_name == 'unet_res34':
        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Unet_res34.pth'))

    elif model_name == 'unet_res101':
        model = smp.Unet(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Unet_res101.pth'))

    elif model_name == 'segformer_mitb0':
        model = smp.Segformer(
            encoder_name="mit_b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Segformer_mitb0.pth'))

    elif model_name == 'segformer_mitb1':
        model = smp.Segformer(
            encoder_name="mit_b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Segformer_mitb1.pth'))

    elif model_name == 'segformer_mitb2':
        model = smp.Segformer(
            encoder_name="mit_b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Segformer_mitb2.pth'))

    elif model_name == 'segformer_mitb3':
        model = smp.Segformer(
            encoder_name="mit_b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        print(model)
        model.load_state_dict(torch.load('/content/PedVision/PedVisionCode/saved_models/Segformer_mitb3.pth'))

    return model

def img_show(image_dir, names, outputs, case_num):
    print(names[case_num])
    # load the image
    image = cv2.imread(image_dir+names[case_num])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30,30))
    for i in range(len(outputs[case_num])):
        plt.subplot(1, len(outputs[case_num]), i+1)
        plt.imshow(image)
        plt.imshow(outputs[case_num][i], alpha=0.7)
        plt.title(f'Class {i}')
        plt.axis('off')
    plt.show()