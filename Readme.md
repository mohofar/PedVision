
# PedVision
Here is the official code for testing and training " Object Level Segmentation for Overcoming Data Scarcity and Variability of Pediatric Images Using Centralized Visual Foundation Model" paper. We would be happy to resolve the issues if you open up an issue here. This is a short video tutorial if you need to follow the steps. The following image shows the pipeline (see the paper for more details) and we go through the codes for each part in the following. 
![pipeline](https://github.com/mohofar/PedVision/blob/main/git_images/Pipeline.jpg)

## Table of Contents
1. [Dependencies Installation](#dependencies-installation)
2. [Training](#training)
3. [Test the pipeline](#test-the-pipeline)
4. [Other consideration](#other-consideration)
5. [License](#license)

## Dependencies Installation
You can use reqirement.txt libraries using `pip install -r requirements.txt`. However, for torch libraries, we used Pytorch version `2.0.0+cu118` and Torchvision version `0.15.1+cu118`. It is recommended to use these versions for full compatibility. 

## Training 
Follow the next checklist for step-by-step training of the pipeline. If you want to just test the pipeline, do the first 5 steps of the checklist and skip others. 

#### ➡️Step1: Initialization 
⬜ After cloning the repo and change the directory to the `PedVision` folder \
⬜ `python PedVisionCode/main.py --foldering y` for foldering\
Use the above comment to construct all related folders for training or fine-tuning the models.

⬜ Put images in `unlabelled_samples`\
For training or testing the pipeline, put all images in the above-mentioned folder. The used image format is PNG, however, changing the format to other formats would not affect the pipeline training a lot. For DiCOM or other medical images, there is a need to add extra lines of code to change them to PNG. Finally, there is no need to resize your images to constant size as this pipeline will do this if needed. 

⬜ Run test script to check compatibility using `ddddddd.py` \
The provided script will check your installation and compatibility for testing or training the code. Continue if all tests are passed.

⬜ Download the trained weights\
All the trained weights of the networks in the last round and foundation models are provided in the next table. We used SAM ViT_h for our pipeline but the other version works too. Download and put them in `PedVisionCode\saved_models` folder. 
| Models  | Weights |
| ------------- | ------------- |
| VFM#1 (SAM ViT_h)  | Download  |
| VFM#2 (SAM ViT_l)  | Download  |
| VFM#3 (SAM ViT_b)  | Download  |
| ROI model (Round=12)  | Download  |
| CLS model (Round=12) | Download  |

#### ➡️Step2: First round annotation and training 
⬜ Do image sampling `unlabelled_samples` from using `python PedVisionCode/main.py --images_sampling y --num_samp 100`\
By `num_samp`, you can choose how many samples you need for the first round of training and also annotation using the provided scripts. The annotation procedure is fast enough. Thus, providing more examples at the first round would help the pipeline to present better results in earlier rounds. However, you can run this comment more than once and provide more samples for annotation for the first round. Please be aware that the longest manual procedure is in the first round and after the first round the automation process will help to speed up the training.

⬜ ROI annotation using `python PedVisionCode/main.py --ROI_annotation y`\
The first and last annotation for the ROI model will be this step. Please use the above comment for image annotation for the ROI model. This script plots an image and requests a point coordinate (x,y) that is in arbitrary space within the region of interest of your project (e.g. hand region for ours). Then, will show three results images that possibly cover the ROI part. you can choose to keep by `y` and to ignore by 'n'. However, for some projects, it might be a case to find ROI. You can use the CovexHull algorithm instead to specify an ROI for your data. 

⬜ Train ROI model using `...`\
⬜ Apply VFM on masked images using `...`\
After training the ROI model, apply VFM to use masked images using ROI model predictions. This step needs enough space based on your project. We can not estimate the exact size as it is dependent on different factors like image size, image details, VFM type, and VFM parameters. However, as a rough example for 10 sample images with a size of (2460,2910), 1.65GB of processed data are saved. Note that this data can be removed after training. 

⬜ CLS annotation using `...`\
By running the above comment, all predicted masks using VFM will visualized with a specified number above them. The model will ask for the slice number for each class. The code considered background and irrelevant predicted parts as a separate class (class 0). Note, if the results missed a part of the image, this is not important for the training of the pipeline as the model just wants to recognize all predicted objects in the image. This step will prepare the images for the classifier network and finally save images as .npy files and labels as .txt files in relevant folders. For the mentioned 10 samples, the occupied size was 352MB. 

⬜ Train CLS model using ...\

The previous step will save the best trained models of the ROI and the classifier for the next rounds.
#### ➡️Step3: Next rounds training 
⬜ HITL for selecting the good cases using ...\

⬜ Preparing for the next round\
⬜ fine-tuning ROI model using ...\
⬜ fine-tuning CLS model using ...


## Test the pipeline
For testing the trained PedVision pipeline, please  use...
### Folder and data structure 
the role of each folder
### test on new samples

## Other consideration

Changing CLS model
Changing the class number
supported images and changing
## License
Information about the project's license.
