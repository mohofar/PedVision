import argparse
import sys
sys.path.append('C:/projects/PedVision')

from PedVisionCode.utils import foldering
from PedVisionCode.utils import sample_images
from PedVisionCode.utils import annotation_roi
from PedVisionCode.utils import train_roi_model
from PedVisionCode.utils import VFM
from PedVisionCode.utils import classifier_annotation
from PedVisionCode.utils import train_cls_model
from PedVisionCode.utils import HITL

def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description='PedVisionCode')

  # Add any command-line arguments you need
  parser.add_argument('--foldering', type=str, default='n', help='y/n to create folders')
  parser.add_argument('--images_sampling', type=str, default='n', help=' y/n sample images from unlabelled_samples folder to ROI image folder and classifier image folder')
  parser.add_argument('--num_samp', type=int, default=2, help='Number of samples to be taken from unlabelled_samples folder for init round and annotation round')
  parser.add_argument('--ROI_annotation', type=str, default='n', help='y/n to run the ROI annotation framework')
  parser.add_argument('--round', type=int, default=0, help='round number starting 0 to n')
  parser.add_argument('--ROI_train', type=str, default='n', help='y/n to train the ROI model')
  parser.add_argument('--apply_VFM', type=str, default='n', help='y/n Apply Foundation model to get masks')
  parser.add_argument('--CLS_annotation', type=str, default='n', help='y/n to run the classifier annotation framework')
  parser.add_argument('--CLS_train', type=str, default='n', help='y/n to train the classification model')
  parser.add_argument('--HITL', type=str, default='n', help='y/n to run the Human-In-The-Loop framework')
  parser.add_argument('--CLS_model_name', type=str, default='MobileNet', help='MobileNet, EffiB1, or EffiB5')
  parser.add_argument('--HITL_num_samples', type=int, default=2, help='Number of samples to be taken from unlabelled_samples folder for HITL round')
  parser.add_argument('--next_round_preparing', type=str, default='n', help='y/n to prepare for next round')


  # Parse the command-line arguments
  args = parser.parse_args()

  # step 1: create folders
  if args.foldering =='y':
    foldering.construct_folders()

  if args.images_sampling =='y':
    sample_images.sample_images(args.num_samp)

  if args.ROI_annotation =='y':
    annotation_roi.main()

  if args.ROI_train =='y':
    print('Training ROI model...')
    train_roi_model.main(args.round)

  if args.apply_VFM =='y':
    print('VFM is running...')
    VFM.main(args.round)
  
  if args.CLS_annotation =='y':
    print('Classifier annotation is running...')
    classifier_annotation.main()

  if args.CLS_train =='y':
    print('Training classifier model...')
    train_cls_model.main(args.round)

  if args.HITL =='y':
    print('Human-In-The-Loop is running...')
    HITL.main(args.HITL_num_samples, args.CLS_model_name, args.round)

if __name__ == "__main__":
  main()