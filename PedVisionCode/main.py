import argparse
import sys
sys.path.append(r'C:\Users\HAG_M\Desktop\PedVision\PedVision')

from PedVisionCode.utils import (
    foldering, sample_images, roi_annotation, train_roi_model, VFM,
    classifier_annotation, train_cls_model, HITL, next_round_preparing, test_cls_model
)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='PedVisionCode Pipeline')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--foldering', type=str, default='n', help='Create folders (y/n)')
    parser.add_argument('--images_sampling', type=str, default='n', help='Sample images for ROI and classifier folders (y/n)')
    parser.add_argument('--num_samp', type=int, default=2, help='Number of samples for initial and annotation rounds')
    parser.add_argument('--ROI_annotation', type=str, default='n', help='Run ROI annotation framework (y/n)')
    parser.add_argument('--round', type=int, default=0, help='Round number (starting from 0)')
    parser.add_argument('--ROI_train', type=str, default='n', help='Train ROI model (y/n)')
    parser.add_argument('--num_epochs_ROI', type=int, default=10, help='Epochs for ROI model training')
    parser.add_argument('--apply_VFM', type=str, default='n', help='Apply foundation model for masks (y/n)')
    parser.add_argument('--CLS_annotation', type=str, default='n', help='Run classifier annotation framework (y/n)')
    parser.add_argument('--CLS_train', type=str, default='n', help='Train classification model (y/n)')
    parser.add_argument('--num_epochs_CLS', type=int, default=10, help='Epochs for classifier model training')
    parser.add_argument('--HITL', type=str, default='n', help='Run Human-In-The-Loop framework (y/n)')
    parser.add_argument('--CLS_model_name', type=str, default='MobileNet', help='Classifier model: MobileNet, EffiB1, or EffiB5')
    parser.add_argument('--HITL_num_samples', type=int, default=2, help='Samples for HITL round')
    parser.add_argument('--prepare_next_round', type=str, default='n', help='Prepare dataset for next round (y/n)')
    parser.add_argument('--fine_tune', type=str, default='n', help='Fine-tune the models (y/n)')
    parser.add_argument('--test_model', type=str, default='n', help='Test classifier model (y/n)')
    parser.add_argument('--img_name', type=str, help='Image name for testing')
    args = parser.parse_args()

    # Execute functions based on arguments
    if args.foldering == 'y':
        foldering.construct_folders()

    if args.images_sampling == 'y':
        sample_images.sample_images(args.num_samp)

    if args.ROI_annotation == 'y':
        roi_annotation.main()

    if args.ROI_train == 'y':
        print('Training ROI model...')
        train_roi_model.main(args.round, args.fine_tune, args.num_epochs_ROI)

    if args.apply_VFM == 'y':
        print('Running VFM...')
        VFM.main(args.round, test=False)

    if args.CLS_annotation == 'y':
        print('Running classifier annotation...')
        classifier_annotation.main(args.num_classes)

    if args.CLS_train == 'y':
        print('Training classifier model...')
        train_cls_model.main(args.round, args.fine_tune, args.num_classes, args.CLS_model_name, args.num_epochs_CLS)

    if args.HITL == 'y':
        print('Running Human-In-The-Loop...')
        HITL.main(args.HITL_num_samples, args.CLS_model_name, args.round, args.num_classes)

    if args.prepare_next_round == 'y':
        print('Preparing for next round...')
        next_round_preparing.main()

    if args.test_model == 'y':
        print('Testing model...')
        VFM.main(round=args.round, test=True)
        test_cls_model.main(rounds=args.round, cls_num=args.num_classes, model_name=args.CLS_model_name, img_name=args.img_name, num_classes=args.num_classes)

if __name__ == "__main__":
    main()
