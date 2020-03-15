import matplotlib.pyplot as plt
import torch
import helper
import glob
import random
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image


def load_transform(train_dir, valid_dir, test_dir):
    """
    Load and transform training, validation and testing dataset
    loaded from data directory.
    
    :param train_dir: Directory for training dataset
    :param valid_dir: Directory for validation dataset
    :param test_dir: Directory for testing dataset
    :return: tuple of data loaders for training, validation, testing data
    """

    train_transform = transforms.Compose([transforms.RandomRotation(60),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             [0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                              shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return (trainloader, validloader, testloader, train_dataset, valid_dataset,
            test_dataset)


def test_model(model, testloader, device):
    """
    Prints accuracy of model against test data set.
    
    :param model: torch model 
    :param testloader: torch DataLoader for test data 
    :param device: torch device (cpu/gpu)
    """

    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Test accuracy: {100 * accuracy / len(testloader):.3f}%")


def train_model(model, hidden_units, criterion, learning_rate, device, epochs,
                trainloader, validloader):
    """
    Train torch model using training data set using hyperparameters defined by
    args
    
    :param model: torch model
    :param hidden_units: Number of 1st level hidden layers of the network
    :param criterion: Loss function
    :param learning_rate: Learning rate
    :param device: torch device (cpu/gpu)
    :param epochs: Number of epochs
    :param trainloader: torch DataLoader for training data
    :param validloader: torch DataLoader for validation data
    """

    for param in model.parameters():
        param.requires_grad = False
        

    model.classifier = Classifier(hidden_units, model.in_features)

    model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # restart grad to zero
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # check loss and accuracy with validation data
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(
                            device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {valid_loss / len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(model, arch, class_to_idx, hidden_units, epochs, path):
    """
    Save the state of model as a checkpoint
    
    :param model: torch model
    :param class_to_idx: Mapping of classes to indices for image dataset
    :param hidden_units: Number of 1st level hidden layer of network
    :param epochs: Number of epochs
    :param path: File path for the checkpoint 
    """

    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())

    model.class_to_idx = class_to_idx

    checkpoint = {'input_size': hidden_units,
                  'arch': arch,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, path)


def load_checkpoint(filepath, gpu_flag):
    """
    Loads a checkpoint.pth file and rebuilds the model

    :param filepath: filepath for checkpoint.pth
    :param gpu_flag: boolean flag for gpu
    :return: model: torch model
    """
    if gpu_flag:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint['arch'] == "alexnet":
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    
    :param image: filepath for image
    :return img_transpose: numpy arr
    """
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    aspect_ratio = img.width/img.height
    if aspect_ratio > 1.0:
        height = 256
        width = int(256 * aspect_ratio)
    else:
        width = 256
        height = int(256 / aspect_ratio)
    img = img.resize((width, height))
    crop_px = 224
    crop_box = (0.5 * (width - crop_px), 0.5 * (height - crop_px), 0.5 * (width + crop_px), 0.5 * (height + crop_px))
    img = img.crop(crop_box)
    
    img = np.array(img)
    img = img / img.max()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean)/std
    
    img_transpose = img.transpose((2, 1, 0))
            
    return img_transpose


def predict(image_path, model, topk, device):
    """ 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    :param image_path: filepath for image
    :param model: torch model
    :param topk: k number of classes with the highest probabilities to return
    :return top_p: 1-d array of probabilities of the top k items
    :return top_index: 1-d array of class index of the top k items
    """
    
    # TODO: Implement the code to predict the class from an image file
    #device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    #process image before pass to model
    img_data = process_image(image_path)
                    
    input = torch.from_numpy(img_data).type(torch.FloatTensor)
    input = input.unsqueeze(0)
    with torch.no_grad():
        input = input.to(device)
        log_ps = model.forward(input)
        ps = torch.exp(log_ps)
        top_p, top_index = ps.topk(topk, dim=1)
        
    return top_p.squeeze(0), top_index.squeeze(0)


class Classifier(nn.Module):

    def __init__(self, hidden_units, pretrained_input_num):
        super().__init__()
        self.fc1 = nn.Linear(pretrained_input_num, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 102)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.log_softmax(self.fc2(x), dim=1)

        return x
