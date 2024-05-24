import random
import shutil
import os


def sample_images(n=10):
    # find all png images
    for i in os.listdir('PedVisionCode/unlabelled_samples/')[:n]:
        shutil.move('PedVisionCode/unlabelled_samples/' + i, 'PedVisionCode/classifier_samples/images/train/' + i)

    for j in os.listdir('PedVisionCode/classifier_samples/images/train/'):
        shutil.copy('PedVisionCode/classifier_samples/images/train/' + j, 'PedVisionCode/ROI_samples/images/train/' + j)

# sample_images(1)