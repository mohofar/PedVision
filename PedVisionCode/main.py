import sys
sys.path.append('C:/projects/PedVision')

from PedVisionCode import foldering
from PedVisionCode import annotation_interface


def main():
    # step 1: create folders
    foldering.construct_folders()

    '''
      step 2: put all the images in the unlabelled_samples folder
      If you already have the masks and labels, put them in the respective folders
      otherwise, you can use the annotation_interface to create the masks and labels for initial training
    '''

    # step 3: run the annotation_framework to create the masks and labels
    annotation_interface()

    # step 4: train the initial ROI model

    # step 5: train the initial classifier model

    # step 6: run the whole pipeline on  the unlabelled_samples folder

    # step 7: Human in the loop

    # step 8: Fine tune the ROI model and classifier model

    
if __name__ == "__main__":
    main()
    