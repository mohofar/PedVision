import random
import shutil
import os


def sample_images(n=10):
    # select n random png images from unlabelled data and transfer them to classifier image folder
    # find all png images
    images = [f for f in os.listdir('PedVisionCode/unlabelled_samples') if f.endswith('.png')]
    # select n random images and move them to classifier image folder
    for i in range(n):
        image = random.choice(images)
        shutil.move('PedVisionCode/unlabelled_samples/' + image, 'PedVisionCode/classifier_samples/images/train/' + image)
    for j in os.listdir('PedVisionCode/classifier_samples/images/train/'):
        shutil.copy('PedVisionCode/classifier_samples/images/train/' + j, 'PedVisionCode/ROI_samples/images/train/' + j)

# sample_images(1)