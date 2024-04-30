from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from torch.utils.data import random_split

    
import segmentation_models_pytorch as smp
import torch


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.png') and os.path.exists(os.path.join(mask_dir, img.replace('.png', '_mask.png')))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '_mask.png'))
        
        # Read the image using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert the image from NumPy array to a PIL Image
        image = Image.fromarray(image)

        # Load the mask and convert it to a PIL Image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 255, 0)  # Convert mask to binary image

        mask = Image.fromarray(mask)


        if self.transform is not None:
            image, mask = self.transform(image, mask)
            
        return image, mask
    
class CustomTransformTrain:
    def __init__(self):
        # Separate resize transforms for image and mask
        self.resize_image = transforms.Resize((256, 256))
        self.resize_mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        
        # # Normalization parameters (example values, adjust as needed)
        # self.mean = [0.485]
        # self.std = [0.229]


    def __call__(self, image, mask):
        # Resize image and mask
        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        # Apply spatial transformations here

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random zoom in/out
        if random.random() > 0.5:
            scale = random.uniform(0.4, 1.2)  # Scale factor
            image = TF.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
            mask = TF.affine(mask, angle=0, translate=(0, 0), scale=scale, shear=0)

        # Random translation
        if random.random() > 0.5:
            max_dx, max_dy = 64, 64  # Max translation
            translations = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
            image = TF.affine(image, angle=0, translate=translations, scale=1, shear=0)
            mask = TF.affine(mask, angle=0, translate=translations, scale=1, shear=0)

        if random.random() > 0.5:
            angle = random.uniform(-90, 90)  # Rotation angle
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Changing brightness
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.3, 1.5)  # Brightness factor
            image = TF.adjust_brightness(image, brightness_factor)

        # Convert to tensor after all PIL image transformations
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalize the image
        # image = TF.normalize(image, self.mean, self.std)

        # Random transparent shape (rectangle) - for image only
        if random.random() > 0.5:
            # Create a transparent rectangle
            rectangle = torch.zeros_like(image)
            alpha = random.uniform(0.0, 0.8)  # Alpha value

            # Define the rectangle dimensions and position
            x1, y1 = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[2] - 1)
            x2, y2 = random.randint(x1, image.shape[1] - 1), random.randint(y1, image.shape[2] - 1)

            rectangle[:, x1:x2, y1:y2] = 1.0  # Set rectangle area to white

            # Blend the rectangle with the image
            image = (1 - alpha) * image + alpha * rectangle

        return image, mask

def to_convex_hull(mask):
    # Convert the mask to a numpy array
    mask_np = mask.numpy()
    # Apply the convex hull transformation
    hull = convex_hull_image(mask_np)

    # Convert back to a tensor
    return torch.from_numpy(hull.astype(np.float32))

def find_bounding_box(mask):
    # Find indices of non-zero elements
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=2)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def apply_bounding_box(mask):
    rmin, rmax, cmin, cmax = find_bounding_box(mask)
    new_mask = torch.zeros_like(mask)
    new_mask[:, rmin:rmax+1, cmin:cmax+1] = 1
    return new_mask

def dice_score(output, target, threshold=0.5):
    output = (output > threshold).float()
    output_flat = output.view(-1)
    target_flat = target.view(-1)
    intersection = (output_flat * target_flat).sum()
    union = output_flat.sum() + target_flat.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_path='PedVisionCode/saved_models/ROI_model_R0.pth', convex=False):
    best_dice_score = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, masks in train_loader:
            # print(inputs.shape, masks.shape)
            # masks = apply_bounding_box(masks)
            if convex:
                for i in range(masks.shape[0]):
                    masks[i,0,:,:] = to_convex_hull(masks[i,0,:,:])
            else:
                masks = masks/255
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        running_dice_score = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                if convex:
                    for i in range(masks.shape[0]):
                        masks[i,0,:,:] = to_convex_hull(masks[i,0,:,:])
                else:
                    masks = masks/255
                outputs = model(inputs)
                dice = dice_score(outputs, masks)  # dice_score function as defined previously
                running_dice_score += dice
        avg_dice_score = running_dice_score / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Dice Score: {avg_dice_score:.4f}')
        torch.save(model.state_dict(), model_path)

def main():
    list_id = []
    for i in glob('PedVisionCode/ROI_samples/images/train/*.png'):
        list_id.append(i)   

    print(len(list_id), list_id[:5])

    # Set paths to your image and mask directories
    image_dir = 'PedVisionCode/ROI_samples/images'
    mask_dir = 'PedVisionCode/ROI_samples/masks'
    custom_transform = CustomTransformTrain()
    dataset_tr = CustomDataset(image_dir+'/train/', mask_dir+'/train/', transform=custom_transform)
    dataset_va = CustomDataset(image_dir+'/valid/', mask_dir+'/valid/', transform=custom_transform)

    print(len(dataset_tr), len(dataset_va))


    train_loader = DataLoader(dataset_tr, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset_va, batch_size=4, shuffle=False)
    print(len(train_loader), len(val_loader))


    # # Load pre-trained U-Net model
    model = smp.Unet(
        encoder_name="efficientnet-b0", # choose encoder, e.g., resnet34, mobilenet_v2, etc.
        encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1, # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1, # model output channels (number of classes in your dataset)
        activation='sigmoid'
    )
    # model = build_unet() 
    print(model)
    # criterion = nn.BCEWithLogitsLoss()
    #use dice loss
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, convex=True)




if __name__ == "__main__":
  main()