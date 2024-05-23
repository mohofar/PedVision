
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
⬜ correct folder of the repo (cd to repo) \
⬜ `python PedVisionCode/main.py --foldering y` for foldering\
⬜ putting images in `unlabelled_samples`\
⬜ running test script to check compatibility \
⬜ download the trained weights\
#### ➡️Step2: First round annotation and training 
⬜ sampling using `python PedVisionCode/main.py --images_sampling y --num_samp 2`\
⬜ ROI annotation using `python PedVisionCode/main.py --ROI_annotation y`\
⬜ check the results of ROI annotation\
⬜ training ROI model using ... \
⬜ applying VFM on masked images\
⬜ CLS annotation using ...\
⬜ training CLS model using ...\
#### ➡️Step3: Next rounds training 
⬜ HITL for selecting the good cases using ...\
⬜ Preparing for the next round\
⬜ fine-tuning ROI model using ...\
⬜ fine-tuning CLS model using ...















➡️1. After cloning the repo, make sure that you are in the correct folder for running the script.

➡️2. Some of the folders do not exist in the repo, thus run `foldering` script using the following comment.
   `python PedVisionCode/main.py --foldering y`
   
➡️3. The images should be stored in the `unlabelled_samples` folder. We use this folder as the pool of data for our training.
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
### Sampling and the first round
➡️4. Use the following comment for sampling a specific number of images for the first round of training.
`python PedVisionCode/main.py --images_sampling y --num_samp 2`
By `num_samp`, you can choose how many samples you need for the first round of training and also annotation using the provided scripts. The annotation procedure is fast enough. Thus, providing more examples at the first round would help the pipeline to present better results in earlier rounds. However, you can run this comment more than once and provide more samples for annotation for the first round. Please be aware that the longest manual procedure is in the first round and after the first round the automation process will help to speed up the training.

➡️5. The first and last annotation for the ROI model will be this step. Please use the following comment for doing annotation for ROI model.
`python PedVisionCode/main.py --ROI_annotation y`

➡️6.  




### first round
ROI annotation 
ROI training 
CLS annotation
CLS training
HITL
Next round preparing
### next rounds
ROI fine-tuning 
CLS fine-tuning 
HITL
Next round preparing

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
