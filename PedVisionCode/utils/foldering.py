import os

def construct_folders():
    folders = [
        'PedVisionCode/classifier_samples',
        'PedVisionCode/classifier_samples/images',
        'PedVisionCode/classifier_samples/images/train',
        'PedVisionCode/classifier_samples/images/valid',
        'PedVisionCode/classifier_samples/labels',
        'PedVisionCode/classifier_samples/labels/train',
        'PedVisionCode/classifier_samples/labels/valid',
        'PedVisionCode/classifier_samples/masks',
        'PedVisionCode/classifier_samples/masks/train',
        'PedVisionCode/classifier_samples/masks/valid',
        'PedVisionCode/classifier_samples/image_ROI',
        'PedVisionCode/classifier_samples/mask_ROI',
        'PedVisionCode/ROI_samples',
        'PedVisionCode/ROI_samples/images',
        'PedVisionCode/ROI_samples/images/train',
        'PedVisionCode/ROI_samples/images/valid',
        'PedVisionCode/ROI_samples/masks',
        'PedVisionCode/ROI_samples/masks/train',
        'PedVisionCode/ROI_samples/masks/valid',
        'PedVisionCode/saved_models',
        'PedVisionCode/unlabelled_samples',
        'PedVisionCode/temp_results',
        'PedVisionCode/new_training',
        'PedVisionCode/test_data',
        'PedVisionCode/test_data/predicted',
        'PedVisionCode/test_data/input',
        'PedVisionCode/test_data/predicted/VFM',
        'PedVisionCode/test_data/prepared'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# if __name__ == '__main__':
#     construct_folders()
