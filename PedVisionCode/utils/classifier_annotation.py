import numpy  as np
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math 
from skimage.transform import rescale, resize, downscale_local_mean
from glob import glob
from IPython.display import clear_output
from tqdm import tqdm


def label_images(num_masks, folder_path, key, cls_num):

    my_label = np.zeros((num_masks))
    indices = []

    for cls in range(cls_num):

        user_input = input("Enter indices (',' sparated) for class {}: ".format(cls+1))
        
        indices = list(map(int, user_input.split(',')))
        #remove empty elements
        print(indices)
        my_label[indices] = cls+1

    # Save the final labels
    # print(np.array(my_label))
    np.savetxt(folder_path+'labels_'+key+'.txt', my_label, fmt='%d')   

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main():


    folder_path = '//department-p.erasmusmc.nl/genr/GENR/Data/medewerkers/Homayounfar_M_098910/train_sam_res/'

    # Get all npy files in the folder
    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    # Create dictionaries to store image and mask files
    image_files = {}
    mask_files = {}

    # Iterate through npy files and categorize them based on the prefix (image or mask)
    for npy_file in npy_files:
        if npy_file.startswith('image_'):
            key = npy_file.replace('image_', '').replace('.npy', '')
            image_files[key] = np.load(os.path.join(folder_path, npy_file))
        elif npy_file.startswith('masks_'):
            key = npy_file.replace('masks_', '').replace('.npy', '')
            mask_files[key] = np.load(os.path.join(folder_path, npy_file))

    # Find matches based on common keys and load them
    for key in tqdm(set(image_files.keys()) & set(mask_files.keys())):

        # check if folder_path+'labels_'+key+'.npy' is not exists
        if(os.path.exists(folder_path+'labels_'+key+'.npy')):
            continue
        image_data0 = image_files[key]
        image_data = resize(image_data0, (128, 128,3),anti_aliasing=False)
        mask_data = mask_files[key]

        print(key)
        print(f"Image Shape: {image_data.shape}, Mask Shape: {mask_data.shape}")
        num_masks = mask_data.shape[2]

        # Calculate the number of rows and columns for the subplot grid
        num_plots = math.ceil(math.sqrt(num_masks))
        # num_plots = num_plots * num_plots  # Ensure a square grid that fits all masks

        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(image_data0)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(image_data0)
        show_anns(s)
        plt.axis('off')
        plt.title(len(masks))
        plt.show() 

        # Plotting
        plt.figure(figsize=(30, 30))  # Adjust the figure size as needed
        for i in range(num_masks):
            plt.subplot(num_plots, num_plots, i + 1)
            plt.imshow(image_data[:, :, 0], cmap='gray')  # Display the image
            plt.imshow(mask_data[:, :, i], alpha=0.4)    # Overlay the mask with transparency
            plt.axis('off')
            plt.title(i)

        plt.tight_layout()
        plt.show()
        
        label_images(num_masks, folder_path, key, cls_num=2)
        clear_output()

if __name__ == "__main__":
    main()