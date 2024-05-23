import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
from torchvision import transforms
import os
from tqdm import tqdm


def apply_transforms(dataset, transforms):
    transformed_images = []
    # for _ in range(3):
    for image in dataset:
        transformed_image = transforms(image)
        transformed_images.append(transformed_image)

    return torch.stack(transformed_images)

def prepare_data_and_model_classifier(rounds, fine_tune, cls_num):
    # Define transformations
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally with probability 0.5
        transforms.RandomRotation(degrees=20),  # Rotate images randomly within ±10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),  # Translate images by 50% in X and Y
        transforms.RandomAffine(degrees=0, scale=(0.4, 1.2)),  # Zoom out/in images
        transforms.ToTensor()  # Convert PIL Image back to tensor
    ])


    # Path to the directory containing the .npy files
    directory_path = 'PedVisionCode/classifier_samples/'

    # Get a list of all final_image files
    train_files = glob(directory_path + 'masks/train/for_cls_net_*.npy')
    valid_files = glob(directory_path + 'masks/valid/for_cls_net_*.npy')


    def load_data(files, mode='train'):
        images_list = []
        labels_list = []
        for file in files:
            images = np.load(file)
            labels = np.loadtxt(directory_path+'labels/'+str(mode)+'/' + os.path.basename(file).replace('npy', 'txt'))

            images_list.append(images)
            labels_list.append(labels)
        return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)


    # Load and concatenate data for training and validation sets
    train_images, train_labels = load_data(train_files, mode='train')
    valid_images, valid_labels = load_data(valid_files, mode='valid')

    # Convert data to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    valid_images = torch.tensor(valid_images, dtype=torch.float32)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long)
    print(train_images.shape, train_labels.shape)
    print(valid_images.shape, valid_labels.shape)

    # Apply transformations to training images
    train_images_transformed = apply_transforms(train_images, train_transforms)

    # Create datasets
    # Create new dataset with transformed images
    train_dataset_transformed = TensorDataset(train_images_transformed, train_labels)
    valid_dataset = TensorDataset(valid_images, valid_labels)

    # Create data loaders
    # Use the transformed dataset in DataLoader
    train_loader = DataLoader(train_dataset_transformed, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    # Load a pre-trained ResNet50 model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, cls_num)  # Assuming cls_num classes

    if rounds>0 and fine_tune=='y':
        model.load_state_dict(torch.load('PedVisionCode\saved_models\CLS_model_R'+str(rounds-1)+'.pth'))
        print('Model loaded: ', 'PedVisionCode\saved_models\CLS_model_R'+str(rounds-1)+'.pth')


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return train_loader, valid_loader, model, criterion, optimizer

def main(rounds, fine_tune, cls_num):
    epochs = 10
    save_path='PedVisionCode/saved_models/CLS_model_R'+str(rounds)+'.pth'

    best_accuracy = 0.0
    
    train_loader, valid_loader, model, criterion, optimizer = prepare_data_and_model_classifier(rounds, fine_tune, cls_num)

    # Training the model
    for epoch in (range(epochs)):  # Number of epochs
        print(f"Epoch {epoch + 1} of "+str(epochs))
        model.train()
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
                
        # Calculate and print the accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print("Accuracy:", accuracy)
        print(classification_report(all_labels, all_preds))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print("Saving best model at epoch: ", epoch)
        
        
    
# if __name__ == "__main__":
#     main(rounds)