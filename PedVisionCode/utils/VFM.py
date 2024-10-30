import os
import pickle
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def final_masks_preparing(masks, image):
    selected_masks = np.zeros(masks[0]['segmentation'].shape)
    for mask in masks:
        selected_masks += mask['segmentation']
    
    whole_mask = resize(selected_masks, (256, 256), anti_aliasing=False)[:, :, np.newaxis].astype(np.bool_)
    for_network = np.zeros((len(masks), 3, 256, 256))
    
    for i, mask in enumerate(masks):
        for_network[i, 0, :, :] = resize(image, (256, 256), anti_aliasing=False)
        for_network[i, 1, :, :] = resize(mask['segmentation'], (256, 256), anti_aliasing=False)
        for_network[i, 2, :, :] = whole_mask[:, :, 0]
    
    return for_network


def tensor_to_image(tensor):
    return tensor.detach().cpu().numpy().squeeze()


class CustomTransformTest:
    def __init__(self):
        self.resize = transforms.Resize((256, 256))

    def __call__(self, image):
        image = self.resize(image)
        return TF.to_tensor(image)


class CustomDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir)
        image = Image.fromarray(np.array(image))
        if self.transform:
            image = self.transform(image)
        return image


def hand_prediction(image_path, model_ROI, show=False):
    dataset = DataLoader(CustomDatasetTest(image_path, transform=CustomTransformTest()), batch_size=1, shuffle=True)
    for image in dataset:
        with torch.no_grad():
            pred = model_ROI(image)
        image_np = tensor_to_image(image)
        pred_np = tensor_to_image(pred)
        if show:
            plt.figure(figsize=(15, 5))
            plt.imshow(image_np, cmap='gray')
            plt.axis('off')
            plt.imshow(pred_np, alpha=0.5)
            plt.axis('off')
            plt.show()
        else:
            return pred_np, image_np
        break


def find_nonzero_boundaries(binary_image):
    non_zero_indices = np.where(binary_image != 0)
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()
    return x_min, x_max, y_min, y_max


def remove_covered_seg(masks_masked):
    fully_covered_indices = []
    for i, mask1 in enumerate(masks_masked):
        for j, mask2 in enumerate(masks_masked):
            if i != j and np.array_equal(mask2['segmentation'], np.logical_and(mask1['segmentation'], mask2['segmentation'])):
                fully_covered_indices.append(i)
    covering_image_indices = list(set(fully_covered_indices))
    return [mask for i, mask in enumerate(masks_masked) if i not in covering_image_indices]


def get_masks(main_path, image_name, model_ROI, mask_generator):
    image = cv2.imread(os.path.join(main_path, image_name))
    original_croped_shape = image.shape
    hand_roi, _ = hand_prediction(os.path.join(main_path, image_name), model_ROI, show=False)
    hand_roi = resize(hand_roi, image.shape[:2], anti_aliasing=False)
    hand_roi = np.where(hand_roi > 0.5, 1, 0)
    x_min, x_max, y_min, y_max = find_nonzero_boundaries(hand_roi)
    image2 = image[y_min:y_max, x_min:x_max]
    hand_roi2 = hand_roi[y_min:y_max, x_min:x_max]
    image2 *= hand_roi2[:, :, np.newaxis]
    masks_masked = mask_generator.generate(image2)
    return [y_min, y_max, x_min, x_max], original_croped_shape, masks_masked, image2


def main(round, test=False):
    images_folder = "PedVisionCode/classifier_samples/images/train/" if not test else "PedVisionCode/test_data/input/"
    results_folder = "PedVisionCode/classifier_sampless/masks/train/" if not test else "PedVisionCode/test_data/predicted/VFM/"
    sam_checkpoint = "PedVisionCode/saved_models/sam_vit_h_4b8939.pth"
    device = "cuda"
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    model_ROI = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation='sigmoid'
    )
    model_ROI.load_state_dict(torch.load(f"PedVisionCode/saved_models/ROI_model_R{round}.pth"))
    model_ROI.eval()

    images_list = [file for file in os.listdir(images_folder) if file.lower().endswith(('.jpg', '.jpeg', 'png'))]
    
    for image_filename in tqdm(images_list):
        mask_filename = f"org_mask_{image_filename[:-4]}.pkl"
        if os.path.exists(os.path.join(results_folder, mask_filename)):
            print(f"Skipping {image_filename} as it already exists")
            continue

        img = Image.open(os.path.join(images_folder, image_filename)).convert('L')
        img1 = np.array(img)[find_nonzero_boundaries(get_masks(images_folder, image_filename, model_ROI, mask_generator)[2])[:2]]
        final_masks = final_masks_preparing(remove_covered_seg(get_masks(images_folder, image_filename, model_ROI, mask_generator)[2]), img1)
        
        np.save(f"{results_folder}for_cls_net_{image_filename[:-4]}.npy", final_masks)
        with open(f"{results_folder}org_mask_{image_filename[:-4]}.pkl", 'wb') as f:
            pickle.dump(final_masks, f)
