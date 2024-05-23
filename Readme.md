# PedVision

Welcome to the official code repository for the paper "Object Level Segmentation for Overcoming Data Scarcity and Variability of Pediatric Images Using Centralized Visual Foundation Model". We are committed to addressing any issues you encounter—please open an issue on this repository if you need assistance. A short video tutorial is also available to help you follow the steps. The image below outlines the pipeline (see the paper for more details), and we will walk you through the code for each part in the sections below.

![pipeline](https://github.com/mohofar/PedVision/blob/main/git_images/Pipeline.jpg)

## Table of Contents
1. [Dependencies Installation](#dependencies-installation)
2. [Training](#training)
3. [Testing the Pipeline](#testing-the-pipeline)
4. [Other Considerations](#other-considerations)
5. [ToDo](#todo)
6. [License](#license)

## Dependencies Installation
To install the necessary libraries, use the `requirements.txt` file:
```
pip install -r requirements.txt
```
For PyTorch libraries, we recommend using PyTorch version `2.0.0+cu118` and Torchvision version `0.15.1+cu118` for full compatibility.

## Training 
Follow the steps below for training the pipeline. If you only want to test the pipeline, complete the first five steps and skip the rest.

### Step 1: Initialization 
1. **Folder Structure Setup:**
```
python PedVisionCode/main.py --foldering y
```
   This command sets up the required folders for training or fine-tuning the models.

2. **Place Images in `unlabelled_samples`:**
   Put all your images in the `unlabelled_samples` folder. The pipeline supports PNG images. For DICOM or other medical image formats, additional code is needed to convert them to PNG. Image resizing is handled by the pipeline as necessary.

3. **Run Compatibility Test:**
```
python ddddddd.py
```
   This script checks your installation and compatibility for testing or training the code. Proceed if all tests pass.

4. **Download Trained Weights:**
   Download and place the trained weights in the `PedVisionCode/saved_models` folder.
   | Model | Weights |
   |-------|---------|
   | VFM#1 (SAM ViT_h) | [Download](#) |
   | VFM#2 (SAM ViT_l) | [Download](#) |
   | VFM#3 (SAM ViT_b) | [Download](#) |
   | ROI model (Round=12) | [Download](#) |
   | CLS model (Round=12) | [Download](#) |

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

3. **Train ROI Model:**
```
python PedVisionCode/main.py --ROI_train y --round 0
```
   Initial training of the ROI model.

4. **Apply VFM on Masked Images:**
```
python PedVisionCode/main.py --apply_VFM y --round 0
```
   This step processes masked images using the trained ROI model. The required storage space varies depending on factors like image size and VFM parameters.

5. **CLS Annotation:**
```
python PedVisionCode/main.py --CLS_annotation y --num_classes 5 --CLS_model_name MobileNet
```
   Annotate the predicted masks for classification. This step prepares images for the classifier network, saving them as `.npy` and `.txt` files.

6. **Train CLS Model:**
```
python PedVisionCode/main.py --ROI_train y --round 0 --num_classes 5
```

### Step 3: Subsequent Rounds Training 
1. **Human-in-the-Loop (HITL) Selection:**
```
python PedVisionCode/main.py --HITL y --CLS_model_name MobileNet --HITL_num_samples 3 --round 0
```
   Validate predicted masks for each class, marking correct predictions with `y` and incorrect ones with `n`.

2. **Prepare for Next Round:**
   Move processed and confirmed images to the appropriate folders, readying the dataset for fine-tuning.

3. **Fine-tune ROI Model:**
```
python PedVisionCode/main.py --ROI_train y --round 1 --fine_tune y
```
4. **Apply VFM on Masked Images:**
```
python PedVisionCode/main.py --apply_VFM y --round 1
```
5. **Fine-tune CLS Model:**
```
python PedVisionCode/main.py --ROI_train y --round 1 --fine_tune y --num_classes 5
```
6. **Repeat Step 3 for Subsequent Rounds:**

## Testing the Pipeline
To test the trained PedVision pipeline, follow the Step 1 checklist, then run:
```
# Add the specific command for testing here
```

## Other Considerations

### Changing CLS Model
To use different networks for the CLS or ROI model, use the `--fine-tuning n` option and modify the code to include your network. Larger networks can improve results in later rounds.

## ToDo
- [ ] Add test dependencies
- [ ] Clone the repo and test on different devices
- [ ] Add recorded video tutorial

## License
This project is licensed under the terms of the MIT license.
