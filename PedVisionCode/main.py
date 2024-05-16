import argparse
import sys
sys.path.append('C:/projects/PedVision')

from PedVisionCode.utils import foldering
from PedVisionCode.utils import sample_images
from PedVisionCode.utils import annotation_roi
from PedVisionCode.utils import train_roi_model
from PedVisionCode.utils import VFM
from PedVisionCode.utils import classifier_annotation


def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description='PedVisionCode')

  # Add any command-line arguments you need
  parser.add_argument('--foldering', type=str, default='n', help='y/n to create folders')
  parser.add_argument('--images_sampling', type=str, default='n', help=' y/n sample images from unlabelled_samples folder to ROI image folder and classifier image folder')
  parser.add_argument('--num_samp', type=int, default=2, help='Number of samples to be taken from unlabelled_samples folder for init round and annotation round')
  parser.add_argument('--ROI_annotation', type=str, default='n', help='y/n to run the ROI annotation framework')
  parser.add_argument('--train_roi_model', type=str, default='n', help='y/n to train the ROI model')
  parser.add_argument('--apply_FoM', type=str, default='n', help='y/n Apply Foundation model to get masks')

  # parser.add_argument('--cls_annotation', type=str, help='y/n to run the classifier annotation framework')


  # Parse the command-line arguments
  args = parser.parse_args()

  # step 1: create folders
  if args.foldering =='y':
    foldering.construct_folders()

  if args.images_sampling =='y':
    sample_images.sample_images(args.num_samp)

  if args.ROI_annotation =='y':
    annotation_roi.main()

  if args.train_roi_model =='y':
    print('Training ROI model is running...')
    train_roi_model.main()

  if args.apply_VFM =='y':
    print('VFM is running...')
    VFM.main()
  
  # if args.cls_annotation =='y':
  #   print('Classifier annotation is running...')
  #   classifier_annotation.main()

if __name__ == "__main__":
  main()