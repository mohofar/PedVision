import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import sys
import shutil
import os
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pickle 

from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from skimage.transform import resize

from PIL import Image
import pickle

def get_masks(main_path, image_name):
    image = cv2.imread(main_path+image_name)
    original_croped_shape = image.shape
    # hand_roi, _ = hand_prediction(main_path+image_name, show=False)
    # #resize hand_roi to image shape
    # hand_roi = resize(hand_roi, (image.shape[0], image.shape[1]),anti_aliasing=False)
    # # make it zero and one  
    # hand_roi[hand_roi > 0.5] = 1
    # hand_roi[hand_roi <= 0.5] = 0
    # # hand_roi, image, angle = orientation_correction(hand_roi, image)
    # x_min, x_max, y_min, y_max = find_nonzero_boundaries(hand_roi)
    # image2 = image[y_min:y_max, x_min:x_max]
    # hand_roi2 = hand_roi[y_min:y_max, x_min:x_max]

    # for i in range(3):
    #     image2[:,:,i] = image2[:,:,i] * hand_roi2

    masks_masked = mask_generator_2.generate(image)

    # return [y_min, y_max, x_min, x_max], original_croped_shape, masks_masked, image2
    return original_croped_shape, masks_masked

def final_masks_preparing(masks, image):
    selected_masks = np.zeros((masks[0]['segmentation'].shape))
    for i in range(len(masks)):
        selected_masks+=masks[i]['segmentation']

    whole_mask = resize(selected_masks, (256, 256),anti_aliasing=False)[:,:, np.newaxis].astype(np.bool_)

    
    image_resized = resize(image, (256, 256),anti_aliasing=False)

    for_network = np.zeros((len(masks),3,256,256))
    for i in range(len(masks)):
        for_network[i,0,:,:] = resize(image, (256, 256),anti_aliasing=False)                        # real image
        for_network[i,1,:,:] = resize(masks[i]['segmentation'], (256, 256),anti_aliasing=False)     # partial mask
        for_network[i,2,:,:] = whole_mask[:,:, 0]                                                   # whole masks

    return for_network


def pipeline(main_path, image_name, masks_save_folder):
  
    img = Image.open(main_path+image_name) #for example image size : 28x28x3
    img1 = np.array(img.convert('L'))  #convert a gray scale
    original_shape, masks = get_masks(main_path, image_name)
    # img1 = img1[crop_limits[0]:crop_limits[1], crop_limits[2]:crop_limits[3]]
    # masks = remove_covered_seg(masks)
    final_masks = final_masks_preparing(masks, img1)
    np.save(masks_save_folder+'for_cls_net_'+image_name[:-4]+'.npy',final_masks)
    # Use 'wb' to write binary data
    with open(masks_save_folder+'org_mask_'+image_name[:-4]+'.pkl', 'wb') as f:
        pickle.dump(masks, f)
    # print(len(masks), final_masks.shape)

def main(main_path, masks_save_folder ,num_new_cases):
    # Get a list of all PNG files in the main_path directory
    image_files = [file for file in os.listdir(main_path) if file.endswith('.png')][:num_new_cases]

    # Process each image file
    for image_name in tqdm(image_files):
        print(image_name)
        pipeline(main_path, image_name, masks_save_folder)
        # break




def main():
    sam_checkpoint = "PedVisionCode/saved_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    # Create a grid to display the images
    images_folder = "PedVisionCode/classifier_samples/images/train/"
    results_folder = "PedVisionCode/classifier_samples/masks/train/"
    for image_filename in tqdm(os.listdir(images_folder)):
        mask_filename = f"{image_filename[:-4]}_mask.pkl"
        if os.path.exists(os.path.join(results_folder, mask_filename)):
            continue

        # Load the image using cv2
        image = cv2.imread(os.path.join(images_folder, image_filename))
        print(image.shape)

        masks = mask_generator.generate(image)
        print(len(masks))

        # save the pkl masks in the results folder  
        mask_filename = f"{image_filename[:-4]}_mask.pkl"
        with open(os.path.join(results_folder, mask_filename), "wb") as f:
            pickle.dump(masks, f)
        

if __name__ == "__main__":
    main()