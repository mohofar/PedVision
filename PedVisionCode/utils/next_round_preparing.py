import os
import pickle
import shutil
import numpy as np
from PIL import Image
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm



def resize_to_original(masks):
    for i in range(len(masks)):
        zero_mask = np.zeros(masks[0]['original_shape'][0:2])
        zero_mask[masks[0]['crop_limits'][0]:masks[0]['crop_limits'][1], masks[0]['crop_limits'][2]:masks[0]['crop_limits'][3]] = masks[i]['segmentation']
        masks[i]['segmentation'] = zero_mask
    return masks

def lbl_generation(masks, image_name):
    txt_lbl = []
    for i in range(len(masks)):
        txt_lbl.append(int(masks[i]['prediction']))
    np.savetxt('PedVisionCode/new_training/for_cls_net_'+image_name+'.txt', txt_lbl,fmt='%d')



def final_masks_preparing(masks, image):
    selected_masks = np.zeros((masks[0]['segmentation'].shape))
    for i in range(len(masks)):
        selected_masks+=masks[i]['segmentation']

    whole_mask = resize(selected_masks, (256, 256),anti_aliasing=False)[:,:, np.newaxis].astype(np.bool_)


    for_network = np.zeros((len(masks),3,256,256))
    for i in range(len(masks)):
        for_network[i,0,:,:] = resize(image, (256, 256),anti_aliasing=False)                        # real image
        for_network[i,1,:,:] = resize(masks[i]['segmentation'], (256, 256),anti_aliasing=False)     # partial mask
        for_network[i,2,:,:] = whole_mask[:,:, 0]                                                   # whole masks

    return for_network

def pipeline_update_training_data(png_dir='PedVisionCode/unlabelled_samples/'):
    # list of all pkl files
    pkl_files = [file for file in os.listdir('PedVisionCode/new_training/') if file.endswith('.pkl')]

    for pkl_file in tqdm(pkl_files):
        try:
            with open('PedVisionCode/new_training/'+pkl_file, 'rb') as f:
                masks = pickle.load(f)

            # load the image
            img = Image.open(png_dir + pkl_file[:-4]+'.png')
            img1 = np.array(img.convert('L'))

            masks = resize_to_original(masks)
            lbl_generation(masks, pkl_file[:-4])
            final_masks = final_masks_preparing(masks, img1)
            np.save('PedVisionCode/new_training/for_cls_net_'+pkl_file[:-4]+'.npy',final_masks)
        except Exception as e:
            print(e)
            continue

def move_files_to_training_folders(png_dir='PedVisionCode/unlabelled_samples/'):
    files_name = [file for file in os.listdir('PedVisionCode/new_training/') if file.endswith('.pkl')]
    for file_name in tqdm(files_name):
        try:
            # Move the image file
            shutil.move(png_dir + file_name[:-4] + '.png', 'PedVisionCode/classifier_samples/images/train/' + file_name[:-4] + '.png')
            
            # Move the mask file
            shutil.move('PedVisionCode/new_training/for_cls_net_' + file_name[:-4] + '.npy', 'PedVisionCode/classifier_samples/masks/train/for_cls_net_' + file_name[:-4] + '.npy')
            
            # Move the label file
            shutil.move('PedVisionCode/new_training/for_cls_net_' + file_name[:-4] + '.txt', 'PedVisionCode/classifier_samples/labels/train/for_cls_net_' + file_name[:-4] + '.txt')
            
            # Move the pkl file
            shutil.move('PedVisionCode/new_training//' + file_name[:-4] + '.pkl', 'PedVisionCode/classifier_samples/masks/train/org_mask_' + file_name)
        except Exception as e:
            print(e)
            continue


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

def process_pkl_files(image_path, mask_path, save_path_image, save_path_mask, predictor):
    pkl_files = [file for file in os.listdir(mask_path) if file.endswith('.pkl')]
    
    for pkl_file in tqdm(pkl_files):
        with open(os.path.join(mask_path, pkl_file), 'rb') as f:
            org_masks = pickle.load(f)

        if 'center' not in org_masks[0].keys():
            continue

        img_file = pkl_file.replace('org_mask_', '')
        img_file = img_file.replace('pkl', 'png')
        img = cv2.imread(os.path.join(image_path, img_file))
        # make a copy of the original image to image path save
        cv2.imwrite(save_path_image + '/' + img_file, img)
        print(img.shape)
        labels = []
        for mask in org_masks:
            labels.append(mask['prediction'])
        labels = np.array(labels)

        hand_tips = np.where(labels == 1)
        masks_of_tips = [org_masks[i] for i in hand_tips[0]]

        predictor.set_image(img)
        input_point = np.zeros((len(masks_of_tips), 2))
        input_label = np.ones(len(masks_of_tips))

        for i in range(len(masks_of_tips)):
            input_point[i] = org_masks[i]['center']

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        print(masks[np.argmax(scores)].shape)
        pkl_file = pkl_file.replace('org_mask_', '')
        cv2.imwrite(save_path_mask + '/' + pkl_file[:-4] + '_mask.png', (masks[np.argmax(scores)] * 255).astype(np.uint8))

def move_ROI_images_masks(in_path, out_path):
    files_name = [file for file in tqdm(os.listdir(in_path)) if file.endswith('.png')]
    for file_name in tqdm(files_name):
        try:
            # Move the image file
            shutil.move(in_path + '/' + file_name, out_path + '/' + file_name)
        except Exception as e:
            print(e)
            continue

def final_check():
    checking = input('Check the \classifier_samples\mask_ROI folder and make sure all the generated masks are correct, if they are not correct, remove both mask and image from the folders.\
                      did it [y/n]?')
    if checking == 'y':
        mask_in_path = 'PedVisionCode/classifier_samples\mask_ROI'
        mask_out_path = 'PedVisionCode/ROI_samples\masks/train'
        image_in_path = 'PedVisionCode/classifier_samples\image_ROI'
        image_out_path = 'PedVisionCode/ROI_samples\images/train'
        move_ROI_images_masks(mask_in_path, mask_out_path)
        move_ROI_images_masks(image_in_path, image_out_path)

def main():
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = "PedVisionCode\saved_models\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)


    pipeline_update_training_data(png_dir='PedVisionCode/unlabelled_samples/')
    move_files_to_training_folders(png_dir='PedVisionCode/unlabelled_samples\/')

    images_path = 'PedVisionCode/classifier_samples\images/train'
    masks_path = 'PedVisionCode/classifier_samples\masks/train'
    labels_path = 'PedVisionCode/classifier_samples\labels/train'

    # load npy file from masks_path

    pkl_files = [file for file in os.listdir(masks_path) if file.endswith('.pkl')]

    # Iterate through each npy file
    for pkl_file in tqdm(pkl_files):

        # print(pkl_file)

        # load pkl file from masks_path
        with open(os.path.join(masks_path, pkl_file), 'rb') as f:
            org_masks = pickle.load(f)
        # if crop_limits and original_shape are not in the pkl file, skip
        if 'crop_limits' not in org_masks[0].keys():
            continue
        if 'center' in org_masks[0].keys():
            continue
            
        print(org_masks[1]['segmentation'].shape)
        print(org_masks[1].keys())
        print(org_masks[1]['crop_box'])
        print(len(org_masks))
        
        # update the name of the file
        txt_file = pkl_file.replace('org_mask_', 'for_cls_net_')
        txt_file = txt_file.replace('pkl', 'txt')
        
        labels = np.loadtxt(os.path.join(labels_path, txt_file))
        print(labels.shape)
        

        hand_tips = np.where(labels == 1)
        print(hand_tips[0])
        # select hand tips from org_masks
        masks_of_tips = [org_masks[i] for i in hand_tips[0]]
        print(len(masks_of_tips))

        # load image file from images_path
        img_file = txt_file.replace('txt', 'png')
        img_file = img_file.replace('for_cls_net_', '')
        print(img_file)

        img = cv2.imread(os.path.join(images_path, img_file))
        print(img.shape)
        print(org_masks[0]['crop_limits'])
        # print(masks_of_tips[0]['segmentation'].shape)
        # extend masks_of_tips to the original size
        for i in range(len(masks_of_tips)):
            zero_mask = np.zeros(org_masks[0]['original_shape'][0:2])
            zero_mask[org_masks[0]['crop_limits'][0]:org_masks[0]['crop_limits'][1], org_masks[0]['crop_limits'][2]:org_masks[0]['crop_limits'][3]] = masks_of_tips[i]['segmentation']
            masks_of_tips[i]['segmentation'] = zero_mask

        # find a central point of one regions for each mask
        for i in range(len(masks_of_tips)):
            mask = masks_of_tips[i]['segmentation']
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX, cY)
            org_masks[i]['center'] = [cX, cY]
        print(org_masks[0].keys())
        # save the pkl file
        with open(os.path.join(masks_path, pkl_file), 'wb') as f:
            pickle.dump(org_masks, f)


    image_path = 'PedVisionCode/classifier_samples\images/train'
    mask_path = 'PedVisionCode/classifier_samples\masks/train'
    save_path_mask = "PedVisionCode/classifier_samples\mask_ROI"
    save_path_image = "PedVisionCode/classifier_samples\image_ROI"

    process_pkl_files(image_path, mask_path, save_path_image, save_path_mask, predictor)
    final_check()


# if __name__ == '__main__':
#     main()