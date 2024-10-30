import argparse
import sys
# sys.path.append(r'C:\Users\HAG_M\Downloads\PedVision-main (1)\PedVision-main')
sys.path.append('/content/PedVision')

from PedVisionCode.utils import foldering
from PedVisionCode.utils import sample_images
from PedVisionCode.utils import roi_annotation
from PedVisionCode.utils import train_roi_model
from PedVisionCode.utils import VFM
from PedVisionCode.utils import classifier_annotation
from PedVisionCode.utils import train_cls_model
from PedVisionCode.utils import HITL
from PedVisionCode.utils import next_round_preparing

from PedVisionCode.utils import test_cls_model

def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description='PedVisionCode')

  # Add any command-line arguments you need
  parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
  parser.add_argument('--foldering', type=str, default='n', help='y/n to create folders')
  parser.add_argument('--images_sampling', type=str, default='n', help=' y/n sample images from unlabelled_samples folder to ROI image folder and classifier image folder')
  parser.add_argument('--num_samp', type=int, default=2, help='Number of samples to be taken from unlabelled_samples folder for init round and annotation round')
  parser.add_argument('--ROI_annotation', type=str, default='n', help='y/n to run the ROI annotation framework')
  parser.add_argument('--round', type=int, default=0, help='round number starting 0 to n')
  parser.add_argument('--ROI_train', type=str, default='n', help='y/n to train the ROI model')
  parser.add_argument('--num_epochs_ROI', type=int, default=10, help='Number of epochs for ROI model training')
  parser.add_argument('--apply_VFM', type=str, default='n', help='y/n Apply Foundation model to get masks')
  parser.add_argument('--CLS_annotation', type=str, default='n', help='y/n to run the classifier annotation framework')
  parser.add_argument('--CLS_train', type=str, default='n', help='y/n to train the classification model')
  parser.add_argument('--num_epochs_CLS', type=int, default=10, help='Number of epochs for classifier model training')
  parser.add_argument('--HITL', type=str, default='n', help='y/n to run the Human-In-The-Loop framework')
  parser.add_argument('--CLS_model_name', type=str, default='MobileNet', help='MobileNet, EffiB1, or EffiB5')
  parser.add_argument('--HITL_num_samples', type=int, default=2, help='Number of samples to be taken from unlabelled_samples folder for HITL round')
  parser.add_argument('--prepare_next_round', type=str, default='n', help='y/n to prepare the dataset for the next round')
  parser.add_argument('--fine_tune', type=str, default='n', help='y/n to fine-tune the models')
  parser.add_argument('--test_model', type=str, default='n', help='y/n to test the classifier model')
  parser.add_argument('--img_name', type=str, help='Image name')
  # Parse the command-line arguments
  args = parser.parse_args()

  # step 1: create folders
  if args.foldering =='y':
    foldering.construct_folders()

  if args.images_sampling =='y':
    sample_images.sample_images(args.num_samp)

  if args.ROI_annotation =='y':
    roi_annotation.main()

  if args.ROI_train =='y':
    print('Training ROI model...')
    train_roi_model.main(args.round, args.fine_tune, args.num_epochs_ROI)

  if args.apply_VFM =='y':
    print('VFM is running...')
    VFM.main(args.round, test=False)
  
  if args.CLS_annotation =='y':
    print('Classifier annotation is running...')
    classifier_annotation.main(args.num_classes)

  if args.CLS_train =='y': 
    print('Training classifier model...')
    train_cls_model.main(args.round, args.fine_tune, args.num_classes, args.CLS_model_name, args.num_epochs_CLS)

  if args.HITL =='y':
    print('Human-In-The-Loop is running...')
    HITL.main(args.HITL_num_samples, args.CLS_model_name, args.round, args.num_classes)

  if args.prepare_next_round =='y': 
    print('Preparing for the next round...')
    next_round_preparing.main()

  if args.test_model =='y':
    print('Testing model...')
    VFM.main(round=args.round, test=True)
    test_cls_model.main(rounds=args.round, cls_num=args.num_classes, model_name=args.CLS_model_name, img_name=args.img_name, num_classes=args.num_classes)

if __name__ == "__main__":
  main()
