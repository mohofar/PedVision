# PedVision

This is the official code repository for the paper "PedVision: Integrating human-in-the-loop with visual foundation models for age-agnostic pediatric hand bone segmentation". The image below outlines the pipeline (see the paper for more details), and we will walk you through the code for each part in the sections below.

![pipeline](https://github.com/mohofar/PedVision/blob/main/git_images/pipeline_.png)

## Table of Contents
1. [Dependencies Installation](#dependencies-installation)
2. [Training](#training)
3. [Testing the Pipeline](#testing-the-pipeline)
4. [Other Considerations](#other-considerations)
5. [ToDo](#todo)
6. [License](#license)

## Dependencies Installation
We used Python 3.10.12 for this project. To install the necessary libraries, use the `requirements.txt` file:
```
pip install -r requirements.txt
```
For PyTorch libraries, we recommend using PyTorch version `2.3.1+cu121` and Torchvision version `0.18.1+cu121` for full compatibility.
Before starting training or testing the pipeline, please update the PATH in `sys.path.append(r'/content/PedVision')
` in main.py file based on your location of the downloaded repo. 
## Training 
Follow the steps below for training the pipeline. If you only want to test the pipeline, complete step 1 and skip the rest.


### Step 1: Initialization 
1. **Folder Structure Setup:**
```
python PedVisionCode/main.py --foldering y
```
   This command sets up the required folders for training or fine-tuning the models.

2. **Place Images in `unlabelled_samples`:**
   Put all your images in the `unlabelled_samples` folder. Note that the pipeline is based on PNG images and for DICOM or other medical image formats, additional code is needed to convert them to PNG. 


3. **Download Trained Weights:**
   Download and place the trained weights in the `PedVisionCode/saved_models` folder.
   | Model | Weights |
   |-------|---------|
   | VFM#1 (SAM ViT_h) | [Download](https://drive.google.com/file/d/1wo75Nv-FrvFtIpm1VeHQTszVrp_inTkL/view?usp=drive_link])|
   | VFM#2 (SAM ViT_l) | [Download](https://drive.google.com/file/d/1nO191wyKFaVoEU6yu9wdVRuZrn-YYU5U/view?usp=drive_link) |
   | VFM#3 (SAM ViT_b) | [Download](https://drive.google.com/file/d/1tRtLQ1Yx1GQ-rardyakvyzg63bbppjD4/view?usp=drive_link)|
   | ROI model (last Round) | [Download](https://drive.google.com/file/d/1qMhAg1cy0s1gJpYOeZWSqvxYraRTTQW-/view?usp=drive_link) |
   | CLS model (last Round) | [Download](https://drive.google.com/file/d/18C3TWfHlCUdBm4cRxUdLij52cJoEZlf6/view?usp=drive_link) |

### Step 2: First Round Annotation and Training 
1. **Image Sampling:**
```
python PedVisionCode/main.py --images_sampling y --num_samp 100
```
   Specify the number of samples for the first round of training and annotation. More samples in the first round can improve early results.

2. **ROI Annotation:**
```
python PedVisionCode/main.py --ROI_annotation y
```
   This step involves annotating the region of interest (ROI). The script requests a point coordinate within the ROI, displays the resulting images, and lets you confirm or reject them. 
   
   **Note for the first round:** Please separate some informative cases (by their ID) from ROI training folders and put them in validation folders. 
   Training folders: `PedVisionCode\ROI_samples\images\train` and `PedVisionCode\ROI_samples\images\valid` 
   Validation folders: `PedVisionCode\ROI_samples\masks\train` and `PedVisionCode\ROI_samples\masks\valid` 
this validations set used to stop and choose the best trained network. Thus the most informative helps the model to be trained more generalized! This step should be done in the first round, then this validation set will be used in the following rounds. 

3. **Train ROI Model:**
```
python PedVisionCode/main.py --ROI_train y --round 0 --num_epochs_ROI 10
```
   Initial training of the ROI model. For the next rounds activate fine_tuning too (used for loading the previous Round trained model) like the following:
```
python PedVisionCode/main.py --ROI_train y --round 0 --num_epochs_ROI 10 --fine_tune y
```

4. **Apply VFM on Masked Images:**
```
python PedVisionCode/main.py --apply_VFM y --round 0
```
   This step processes masked images using the trained ROI model. The required storage space varies depending on factors like image size and VFM parameters.


5. **CLS Annotation:**
```
python PedVisionCode/main.py --CLS_annotation y --num_classes 5 
```
Annotate the predicted masks for classification. This step prepares images for the classifier network, saving them as `.npy` and `.txt` files.

**Note for the first round:** Please separate some validation cases (by their ID) before the first training of the classifier too. Again use the most informative images to cover the possible variations. 
Training folders: `PedVisionCode\classifier_samples\image\train`, `PedVisionCode\classifier_samples\labels\train`, and `PedVisionCode\classifier_samples\masks\train` 
Validation folders: `PedVisionCode\classifier_samples\image\valid`, `PedVisionCode\classifier_samples\labels\valid`, and `PedVisionCode\classifier_samples\masks\valid` 

6. **Train CLS Model:**
```
python PedVisionCode/main.py --CLS_train y --round 0 --num_classes 5 --num_epochs_CLS 5 --CLS_model_name EffiB5
```
CLS_model_name can be `MobileNet`for MobileNetV2, `EffiB1` for EfficientNetB1 and `EffiB5` for EfficientNetB5. Please activate fine_tuning for the next rounds as follows: 
```
python PedVisionCode/main.py --CLS_train y --round 1 --num_classes 5 --num_epochs_CLS 5 --CLS_model_name EffiB5 --fine_tune y
```

### Step 3: Subsequent Rounds Training 
1. **Human-in-the-Loop (HITL) Selection:**
Please copy some unseen images to `unlabelled_samples` to be used for this step. An example comment is mentioned below: 
```
python PedVisionCode/main.py --HITL y --CLS_model_name EffiB5 --HITL_num_samples 20 --round 0 --num_classes 5
```
   Validate predicted masks for each class, marking correct predictions with `y` and incorrect ones with `n`. Using `HITL_num_samples` you can specify how many samples you want to analyze. 

2. **Prepare for Next Round:**
   If you want to continue the training for the next round, do this step to prepare and move your confirmed samples to related folders using the following comment:
   ```
   python PedVisionCode/main.py --prepare_next_round y
   ```

3. **Repeat Step 2 and 3 for Subsequent Rounds**

## Testing the Pipeline
To test the trained PedVision pipeline, follow the Step 1 checklist. Just put your images in `test_data\input\` folder. After first running the following comment, all processed images will be saved in related folders and the next time you run it, you will get the visualization result very fast for each `img_name`you select. The following is an example of testing on the `8992.png` image.
```
python PedVisionCode/main.py  --round 1 --num_classes 5 --test_model y --img_name 8992
```

## ToDo
- [ ] Add recorded video tutorial
- [ ] Colab demo

## License
This project is licensed under the terms of the MIT license.
