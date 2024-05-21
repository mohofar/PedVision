import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.pyplot as plt
import shutil
import os
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def main():
    sam_checkpoint = "PedVisionCode/saved_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # Create a grid to display the images
    images_folder = "PedVisionCode/ROI_samples/images/train"
    results_folder = "PedVisionCode/ROI_samples/masks/train"
    for image_filename in os.listdir(images_folder):
        mask_filename = f"{image_filename[:-4]}_mask.png"
        if os.path.exists(os.path.join(results_folder, mask_filename)):
            continue

        # Clear the output of ipython cell to be ready for the next image
        plt.figure(figsize=(10, 10))

        # Load the image using cv2
        image = cv2.imread(os.path.join(images_folder, image_filename))
        print(image.shape)

        # Display the image in the grid
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Memorize an arbitrary X,Y coordinates within your desire ROI then close this window", fontsize=14)
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)

        # Plot vertical lines each 50 pixels
        for i in range(0, image.shape[1], 50):
            if i % 100 == 0:
                plt.axvline(i, color='r', linestyle='--', linewidth=0.5)
            else:
                plt.axvline(i, color='g', linestyle='--', linewidth=0.5)

        # Plot horizontal lines each 50 pixels
        for i in range(0, image.shape[0], 50):
            if i % 100 == 0:
                plt.axhline(i, color='r', linestyle='--', linewidth=0.5)
            else:
                plt.axhline(i, color='g', linestyle='--', linewidth=0.5)

        plt.show()

        # Get user input for input_point
        x = int(input("Enter the x-coordinate for input_point: "))
        y = int(input("Enter the y-coordinate for input_point: "))
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True)
        print(masks.shape, scores.shape, logits.shape)

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Is the blue region show your ROI [y/n]? close this for answering.", fontsize=14)
            plt.axis('off')
            plt.show()

            x = input('Save mask? (y/n): ')
            if x == 'y':
                cv2.imwrite(os.path.join(results_folder, mask_filename), mask * 255)
            else:
                continue
        clear_output(wait=True)

# if __name__ == "__main__":
#     main()
