import numpy  as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

def plot_for_annotation(data):
    # Extract the first channel of each image
    first_channel_data = data[:, 1, :, :]
    second_channel_data = data[:, 0, :, :]
    # print(first_channel_data.shape)

    # Calculate the number of rows and columns for the grid
    num_images = len(first_channel_data)
    num_rows = int(np.ceil(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 30))

    # Iterate through each image and plot the first channel
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_images:
                axes[i, j].imshow(second_channel_data[index])
                axes[i, j].imshow(first_channel_data[index], alpha=0.5)
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Image {index}')

    # Adjust layout and display the plot
    plt.subplots_adjust(wspace=0.5, hspace=0.5)        
    plt.show()

def label_images(data, npy_file, mode='train'):
    my_label = np.zeros(data.shape[0])
    cls_num = 4
    indices = []

    for cls in range(cls_num):

        user_input = input("Enter indices (',' sparated) for class {}: ".format(cls+1))
        
        indices = list(map(int, user_input.split(',')))
        #remove empty elements
        print(indices)
        my_label[indices] = cls+1

    # Save the final labels
    # print(np.array(my_label))
    np.savetxt('PedVisionCode/classifier_samples/labels/'+str(mode)+'/'+npy_file[:-4]+'.txt', np.array(my_label), fmt='%d')


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


    folder_path = "PedVisionCode/classifier_samples/masks/"

    mode = 'train' #train or valid
    # Get a list of all npy files in the folder
    npy_files = [file for file in os.listdir(folder_path+str(mode)+'/') if file.endswith('.npy')]

    # Iterate through each npy file
    for npy_file in tqdm(npy_files):
        # Skip if the file exists
        if os.path.exists('PedVisionCodee/classifier_samples/labels/'+str(mode)+'/'+npy_file[:-4]+'.txt'):
            continue
        else:
            # Load the npy file
            data = np.load(os.path.join(folder_path+str(mode)+'/', npy_file))
            print(npy_file)

            plot_for_annotation(data)
            label_images(data, npy_file, mode)
            clear_output()


if __name__ == "__main__":
    main()