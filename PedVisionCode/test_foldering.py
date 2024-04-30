import os
import unittest
from foldering import construct_folders

class TestFoldering(unittest.TestCase):
    def test_construct_folders(self):
        # Call the function to create the folders
        construct_folders()

        # Check if the folders are created
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
            'PedVisionCode/unlabelled_samples'
        ]

        for folder in folders:
            self.assertTrue(os.path.exists(folder))

if __name__ == '__main__':
    unittest.main()