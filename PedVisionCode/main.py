import sys
sys.path.append('C:/projects/PedVision')

from PedVisionCode import foldering
from PedVisionCode import annotation_interface


def main():
    # step 1: create folders
    foldering.construct_folders()

    '''
      step 2: put all the images in the unlabelled_samples folder
      If you have already have the masks and labels, put them in the respective folders
      otherwise, you can use the annotation_interface to create the masks and labels
    '''

    # step 3: run the annotation_framework to create the masks and labels
    annotation_interface()

    
if __name__ == "__main__":
    main()
    