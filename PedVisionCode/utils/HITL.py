from torch.utils.data import random_split
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


from skimage.morphology import convex_hull_image
from skimage.transform import resize


import matplotlib.pyplot as plt
from torch.utils.data import random_split


import sys

import segmentation_models_pytorch as smp
import torch


from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 

# sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from skimage.transform import resize
from torchvision import models, transforms
import pickle
from tqdm import tqdm

# Function to convert tensor to numpy image
def tensor_to_image(tensor):
    return tensor.detach().cpu().numpy().squeeze()

class CustomTransformTest:
    def __init__(self):
        self.resize = transforms.Resize((256, 256))
        # Normalization parameters for grayscale images
        # self.mean = [0.485]  # Adjust based on your dataset
        # self.std = [0.229]   # Adjust based on your dataset

    def __call__(self, image):
        # Resize image
        image = self.resize(image)

        # Convert to tensor after all PIL image transformations
        image = TF.to_tensor(image)

        # Normalize the image
        # image = TF.normalize(image, self.mean, self.std)

        return image

class CustomDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = image_dir
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.image_dir, self.images[idx])
        # print(img_path)
        # Read the image using OpenCV
        # image = cv2.imread(self.image_dir, cv2.IMREAD_GRAYSCALE)
        image = Image.open(self.image_dir)
        image = np.array(image)
        # Convert the image from NumPy array to a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)  # Apply transform only to the image
            
        return image

def hand_prediction(image_path, model_ROI, show=False):
    custom_transformtest = CustomTransformTest()
    print(image_path)
    dataset = CustomDatasetTest(image_path, transform=custom_transformtest)
    dataset2 = DataLoader(dataset, batch_size=1, shuffle=True)

    for image in dataset2:
        with torch.no_grad():
            pred = model_ROI(image)

        # Convert to numpy arrays for visualization
        image_np = tensor_to_image(image)
        # mask_np = tensor_to_image(mask)
        pred_np = tensor_to_image(pred)
        if show:
            # Visualization
            plt.figure(figsize=(15, 5))
            # plt.subplot(1, 2, 1)
            plt.imshow(image_np, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            # plt.subplot(1, 2, 2)
            plt.imshow(pred_np, alpha=0.5)
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.show()
        else:
            return pred_np, image_np
        break



def get_masks(main_path, image_name, model_ROI,mask_generator_2):
    image = cv2.imread(main_path+image_name)
    original_croped_shape = image.shape
    hand_roi, _ = hand_prediction(main_path+image_name, model_ROI, show=False)
    #resize hand_roi to image shape
    hand_roi = resize(hand_roi, (image.shape[0], image.shape[1]),anti_aliasing=False)
    # make it zero and one  
    hand_roi[hand_roi > 0.5] = 1
    hand_roi[hand_roi <= 0.5] = 0
    # hand_roi, image, angle = orientation_correction(hand_roi, image)
    x_min, x_max, y_min, y_max = find_nonzero_boundaries(hand_roi)
    image2 = image[y_min:y_max, x_min:x_max]
    hand_roi2 = hand_roi[y_min:y_max, x_min:x_max]

    for i in range(3):
        image2[:,:,i] = image2[:,:,i] * hand_roi2

    masks_masked = mask_generator_2.generate(image2)

    return [y_min, y_max, x_min, x_max], original_croped_shape, masks_masked, image2


def find_nonzero_boundaries(binary_image):
    """
    Finds the boundaries of non-zero pixels in a binary image.

    Parameters:
    binary_image (numpy.ndarray): A binary image where non-zero pixels represent the features of interest.

    Returns:
    tuple: The minimum and maximum indices in the x and y directions.
    """
    # Find the indices where the array is non-zero
    non_zero_indices = np.where(binary_image != 0)

    # Find the boundaries
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()

    return (x_min, x_max, y_min, y_max)

def remove_covered_seg(masks_masked):
    fully_covered_indices = []

    # Compare each pair of images
    for i in range(len(masks_masked)):
        for j in range(len(masks_masked)):
            if i != j:
                image1 = masks_masked[i]['segmentation']
                image2 = masks_masked[j]['segmentation']

                # Check if image1 fully covers image2
                if np.array_equal(image2, np.logical_and(image1, image2)):
                    fully_covered_indices.append(i)

    # List of covering_image_index
    covering_image_indices = list(set(fully_covered_indices))
    masks_masked = [image for index, image in enumerate(masks_masked) if index not in covering_image_indices]

    return masks_masked

def final_masks_preparing(masks, image):
    selected_masks = np.zeros((masks[0]['segmentation'].shape))
    for i in range(len(masks)):
        selected_masks+=masks[i]['segmentation']

    whole_mask = resize(selected_masks, (256, 256),anti_aliasing=False)[:,:, np.newaxis].astype(np.bool_)

    
    # image_resized = resize(image, (256, 256),anti_aliasing=False)

    for_network = np.zeros((len(masks),3,256,256))
    for i in range(len(masks)):
        for_network[i,0,:,:] = resize(image, (256, 256),anti_aliasing=False)                        # real image
        for_network[i,1,:,:] = resize(masks[i]['segmentation'], (256, 256),anti_aliasing=False)     # partial mask
        for_network[i,2,:,:] = whole_mask[:,:, 0]                                                   # whole masks

    return for_network

def load_model_and_predict(mask, model_name ,model_path):
    # Load the image using PIL
    # input_image = Image.open(input_path)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    # Initialize the model structure
    if(model_name == 'MobileNet'):
        model_cls = models.mobilenet_v2(weights=None)
        model_cls.classifier[1] = nn.Linear(model_cls.last_channel, 5)
    elif(model_name == 'EffiB1'):
        model_cls = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_features = model_cls.classifier[-1].in_features
        model_cls.classifier[-1] = nn.Linear(in_features, 5)
    elif(model_name == 'EffiB5'):
        
        model_cls = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        in_features = model_cls.classifier[-1].in_features
        model_cls.classifier[-1] = nn.Linear(in_features, 5)
    else:
        print('Specify the model name')
    # model_cls = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
    # in_features = model_cls.classifier[-1].in_features
    # model_cls.classifier[-1] = nn.Linear(in_features, 5)
    

    model_cls.load_state_dict(torch.load(model_path)) # previous version


    model_cls.eval()  # Set the model to evaluation mode

    # Make a prediction
    with torch.no_grad():
        outputs = model_cls(mask)
        _, predicted = torch.max(outputs, 1)

    return predicted

def pipeline(main_path, image_name, image_num, model_name, model_path, model_ROI,mask_generator_2):
    plt.figure(figsize=(10,10))

    img = Image.open(main_path+image_name)
    img1 = np.array(img.convert('L'))  
    crop_limits, original_shape, masks, _ = get_masks(main_path, image_name, model_ROI,mask_generator_2)
    img1 = img1[crop_limits[0]:crop_limits[1], crop_limits[2]:crop_limits[3]]
    masks = remove_covered_seg(masks)
    final_masks = final_masks_preparing(masks, img1)
    prediction = load_model_and_predict(final_masks, model_name, model_path)
    for bone in [1,2,3,4]: #should be changed based on the number of classessssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
        overal_mask = np.zeros((masks[0]['segmentation'].shape))
        for i in range(len(prediction)):
            if(prediction[i].numpy() == bone):
                overal_mask = overal_mask + masks[i]['segmentation']
        

        plt.subplot(1,4,bone)
        plt.imshow(img1)
        plt.imshow(overal_mask.astype(np.bool_), alpha=0.7)
        plt.title(image_num)
    plt.show()
    return prediction, masks, crop_limits, original_shape

def main(num_new_cases, model_name,  round):
    main_path = 'PedVisionCode/unlabelled_samples/'
    CLS_model_path = 'PedVisionCode\saved_models\CLS_model_R'+str(round)+'.pth'
    ROI_model_path = 'PedVisionCode\saved_models\ROI_model_R'+str(round)+'.pth'
    sam_checkpoint = "PedVisionCode\saved_models\sam_vit_h_4b8939.pth" 
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    # Load pre-trained U-Net model
    model_ROI = smp.Unet(
        encoder_name="efficientnet-b0", # choose encoder, e.g., resnet34, mobilenet_v2, etc.
        encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1, # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1, # model output channels (number of classes in your dataset)
        activation='sigmoid'
    )
    # Load the saved model weights
    model_ROI.load_state_dict(torch.load(ROI_model_path))
    model_ROI.eval()  # Set the model to evaluation mode
    #Get a list of all PNG files in the main_path directory
    image_files = [file for file in os.listdir(main_path) if file.endswith('.png')]
    #suffel the list
    # random.shuffle(image_files)
    image_files = image_files[:num_new_cases]
    # Process each image file
    list_image_names = []
    list_good_cases = []
    for image_num, image_name in tqdm(enumerate(image_files)):
        prediction, masks, crop_limits, original_shape = pipeline(main_path, image_name, image_num, model_name, CLS_model_path, model_ROI ,mask_generator_2)
        # add prediction, crop_limits, original_shape to masks pkl file 

        for i in range(len(masks)):
            masks[i]['prediction'] = prediction[i].numpy()
            masks[i]['crop_limits'] = crop_limits
            masks[i]['original_shape'] = original_shape

        # save the final masks list
        with open('PedVisionCode/temp_results/'+str(image_name[:-4])+'.pkl', 'wb') as f:
            pickle.dump(masks, f)
        list_image_names.append(str(image_name[:-4])+'.pkl')

        user_input = input("Was it a good prediction[y/n]? ")
        if user_input == 'y':
            list_good_cases.append(int(image_num))
    
    user_input = np.zeros(num_new_cases)
    user_input[list_good_cases] = 1
    for i in range(len(list_image_names)):
        try:
            if(user_input[i] == 0):
                os.remove('PedVisionCode/temp_results/'+list_image_names[i])
            else:
                os.rename('PedVisionCode/temp_results/'+list_image_names[i], 'PedVisionCode/new_training/'+list_image_names[i])
        except Exception as e:
            print(e)
            pass

if __name__ == "__main__":
    # num_new_cases = 2
    # model_name = 'MobileNet'
    main(num_new_cases, model_name, rounds)