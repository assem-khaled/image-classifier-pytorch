# Imports here
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim

import argparse
from collections import OrderedDict
import time


def transfrom(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    # Define the transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomRotation(45),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_val_transform = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    val_data = datasets.ImageFolder(valid_dir, transform = test_val_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_val_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return trainloader, valloader, testloader, train_data.class_to_idx

def build_model(arch, hidden_units, learning_rate, class_idx):
    # loading a pre-trained network
    if arch.lower() == 'vgg19':
        model = models.vgg19(pretrained = True)
        model.name = 'vgg19'    
    elif arch.lower() == 'vgg16':
        model = models.vgg16(pretrained = True)
        model.name = 'vgg16'        
    elif arch.lower() == 'vgg13':
        model = models.vgg13(pretrained = True)
        model.name = 'vgg13'
    else:
        model = models.vgg11(pretrained = True)
        model.name = 'vgg11'
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replacing the pretrained classifier
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                   ('relu', nn.ReLU()),
                                   ('dropout', nn.Dropout(0.5)),
                                   ('fc2', nn.Linear(hidden_units, len(class_idx))),
                                   ('output', nn.LogSoftmax(dim=1)) ]))
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device);
    
    print(model)
    print('GPU: {}, Learning rate: {}'.format(torch.cuda.is_available(), learning_rate))
    
    return model, criterion, optimizer

def train(data_dir, arch, hidden_units, learning_rate, epochs, test=False):
    # applying transforms and loading the dataset
    trainloader, valloader, testloader, class_idx = transfrom(data_dir)
    print('Dataset loaded')
    # building the model
    model, criterion, optimizer = build_model(arch, hidden_units, learning_rate, class_idx)
    # training the model
    print_every = 50
    print('start training for {} epochs'.format(epochs))
    for epoch in range(epochs):
        tic = time.time()
        running_loss = 0
        for i,(images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            #forward pass
            output = model(images)
            loss = criterion(output, labels)

            #backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % print_every == 0:
                val_loss = 0
                val_total = 0
                val_correct = 0
                model.eval()

                with torch.no_grad():
                    for i2, (images, labels) in enumerate(valloader):
                        images, labels = images.to(device), labels.to(device)

                        output = model(images)
                        loss = criterion(output, labels)
                        val_loss += loss.item()

                        # calculate the accuracy
                        predictions = output.max(dim=1)[1]
                        val_correct += (labels.data == predictions).sum().item()
                        val_total += labels.size(0)

                        # the last batch in the validation set is only 50 images not 64, so i will discard the last batch
                        if (i2+1) == len(valloader)-1:
                            toc = time.time()
                            print('Epochs [{}/{}], Step [{}/{}], Avg Trian Loss: {:.3f}, Avg Val Loss: {:.3f}, Avg Val Accuracy: {:.2f}%, Epoch Time taken: {:.2f}'.format(
                            epoch+1, 
                            epochs, 
                            i+1, 
                            len(trainloader), 
                            running_loss/(print_every), 
                            val_loss/(len(valloader)-1), 
                            100*val_correct/val_total, 
                            toc-tic))
                            break

                running_loss = 0
                model.train()
                
    # Do validation on the test set
    if test == True:
        print("testing model's performance..")
        test_correct = 0
        test_total = 0
        tic = time.time()

        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, prediction = torch.max(outputs.data, 1)
                test_correct += (prediction == labels).sum().item()
                test_total += labels.size(0)

        model.train()
        toc = time.time()
        print('Test Accuracy: {:.4f}% \nTime taken: {:.3f} sec'.format(100*test_correct/test_total, toc-tic))

    # save the mapping of classes to indices
    model.class_to_idx = class_idx     
    return model

def save_model(model, model_save_dir, checkpoint=False):
    if checkpoint == True: # saving checkpoint
        checkpoint = {'classifier': model.classifier,
                    'state_dict': model.state_dict(),
                    'mapping': model.class_to_idx,
                    'name': model.name}

        torch.save(checkpoint, model_save_dir)
        print('Checkpoint saved')
    else: # saving model
        torch.save(model, model_save_dir)
        print('Model saved')


    
    
def main():
    #initialize the parser
    parser = argparse.ArgumentParser(description='to train the model type: python train.py <data_directory> --model_save_dir <model_save_directory> --checkpoint <to save only a checkpoint> --arch <architecture vgg11, vgg13, vgg16 or vgg19> --learning_rate <float_num> --hidden_units <int_num> --epochs <int_num> --gpu --test')

    #Add the positional parameters
    parser.add_argument('data_directory', help='Path to the dataset', type=str)
    #Add the optional parameters
    parser.add_argument('--model_save_dir', help='Path to save the model', type=str, default='saved_model.pth')
    parser.add_argument('--checkpoint', help='save model checkpoint', default=False, action='store_true')
    parser.add_argument('--arch', help='Choose vgg architecture', type=str, default='vgg11')
    parser.add_argument('--learning_rate', help='Choose learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', help='Number of hidden units', type=int, default=512)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=20)
    parser.add_argument('--gpu', help='Enable gpu', default=False, action='store_true')
    parser.add_argument('--test', help='test model on test dataset', default=False, action='store_true')
    
    #Parse the argument
    args = parser.parse_args()
    # setting the device
    global device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print('using:',device)
    # training the model
    model = train(args.data_directory, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.test)
    # save model to directory
    save_model(model, args.model_save_dir, args.checkpoint)


if __name__ == '__main__':
    main()