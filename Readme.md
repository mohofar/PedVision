# PedVision

This is the official code repository for the paper "PedVision: Integrating human-in-the-loop with visual foundation models for age-agnostic pediatric hand bone segmentation". The image below outlines the pipeline (s: ee the paper for more details), and we will walk you through the code for each part in the sections below.

![pipeline](https://github.com/mohofar/PedVision/blob/main/git_images/pipeline_.png)

## Table of Contents
1. [Dependencies Installation](#dependencies-installation)
2. [Testing the Pipeline](#testing-the-pipeline)
3. [Training the Pipeline](#training-the-pipeline)
4. [License](#license)
   
> [!NOTE]
> - Open this link in a new tab to see toturial on YouTube for test part: [PedVision test](https://youtu.be/hi7YJ2_5c7U?si=YMx7D0kBZptvXszS)
> - Training toturial will be avaliable soon.

## Dependencies Installation
Follow the following steps to install all dependencies!

```
conda create -n pedvision_env python=3.10.12 
conda activate pedvision_env
git clone https://github.com/mohofar/PedVision.git
cd PedVision
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Please update the PATH in `sys.path.append(r'./PedVision')
` in main.py file based on your location of the downloaded repo. 



## Testing the pipeline
To test the trained PedVision pipeline, refer to the following example using the `1512.png` image.
```
python PedVisionCode/main.py --foldering y
```
**Download Trained Weights:**
   Download and place the trained weights in the `PedVisionCode/saved_models` folder.
   | Model | Weights |
   |-------|---------|
   | VFM#1 (SAM ViT_h) | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)|
   | ROI model (last Round) | [Download](https://drive.google.com/file/d/1K0PphrPPlv3mmlW1dIVcqhA2V8ZyVe5u/view?usp=drive_link) |
   | CLS model (last Round) | [Download](https://drive.google.com/file/d/17q_-KDSkWPPItRgj9a-knF82-ZiFXUcQ/view?usp=drive_link) |
```
python PedVisionCode/main.py --test_model y --round 11 --num_classes 5  --img_name 1512 --CLS_model_name EffiB5
```
**Note**: The pipeline will prepare the masks of all images in the test folder in the first run.
## Training the pipeline
coming soon... 

## License
This project is licensed under the terms of the MIT license.
