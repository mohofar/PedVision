import sys
sys.path.append('C:/projects/PedVision')

import unittest
import os

# check if we have everything we need to run the tests
class TestInit(unittest.TestCase):
    def test_image_loading(self):

        # check if the pth and png files are available
        files = [
            # 'PedVisionCode/classifier_samples/images/train/1.pth',
            # 'PedVisionCode/classifier_samples/images/train/1.png',
            # 'PedVisionCode/classifier_samples/images/valid/1.pth',
            # 'PedVisionCode/classifier_samples/images/valid/1.png',
            # 'PedVisionCode/ROI_samples/images/train/1.pth',
            # 'PedVisionCode/ROI_samples/images/train/1.png',
            # 'PedVisionCode/ROI_samples/images/valid/1.pth',
            # 'PedVisionCode/ROI_samples/images/valid/1.png',
            'PedVisionCode/saved_models/sam_vit_h_4b8939.pth'

        ]

        for file in files:
            self.assertTrue(os.path.exists(file))

if __name__ == '__main__':
    unittest.main()