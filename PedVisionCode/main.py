import argparse
import sys
sys.path.append('C:/projects/PedVision')

from PedVisionCode.utils import foldering
from PedVisionCode.utils import sample_images
from PedVisionCode import annotation_interface



def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description='PedVisionCode')

  # Add any command-line arguments you need
  parser.add_argument('--foldering', type=str, help='y/n to create folders')
  parser.add_argument('--images_sampling', type=str, help=' y/n sample images from unlabelled_samples folder to ROI image folder and classifier image folder')
  parser.add_argument('--Num_samp', type=int, help='Number of samples to be taken from unlabelled_samples folder for init round and annotation round')


  # Parse the command-line arguments
  args = parser.parse_args()

  # step 1: create folders
  if args.foldering =='y':
    foldering.construct_folders()

  if args.images_sampling =='y':
    sample_images.sample_images(args.Num_samp)

  # step 3: run the annotation_framework to create the masks and labels
  # providing random samples from unlabelled_samples folder to ROI image folder
  # annotation_interface()

  # # Perform the specified steps based on the command-line arguments
  # if args.train_roi:
  #   # step 4: train the initial ROI model
  #   train_roi_model()

  # if args.train_classifier:
  #   # step 5: train the initial classifier model
  #   train_classifier_model()

  # if args.run_pipeline:
  #   # step 6: run the whole pipeline on the unlabelled_samples folder
  #   run_pipeline()

  # if args.human_in_the_loop:
  #   # step 7: Human in the loop
  #   human_in_the_loop()

  # if args.fine_tune:
  #   # step 8: Fine tune the ROI model and classifier model
  #   fine_tune_models()


# def train_roi_model():
#   # Code to train the initial ROI model
#   pass


# def train_classifier_model():
#   # Code to train the initial classifier model
#   pass


# def run_pipeline():
#   # Code to run the whole pipeline on the unlabelled_samples folder
#   pass


# def human_in_the_loop():
#   # Code for the Human in the loop step
#   pass


# def fine_tune_models():
#   # Code to fine tune the ROI model and classifier model
#   pass


if __name__ == "__main__":
  main()