# Imports here
from torchvision import transforms, models
import torch

from PIL import Image
import json
import argparse

def load_model(model_path , checkpoint = False):
    load_point = torch.load(model_path)
    
    if checkpoint == True: # Loading the checkpoint
        if load_point['name'] == 'vgg19':
            my_model = models.vgg19(pretrained = True)
        elif load_point['name'] == 'vgg16':
            my_model = models.vgg16(pretrained = True)    
        elif load_point['name'] == 'vgg13':
            my_model = models.vgg13(pretrained = True)
        elif load_point['name'] == 'vgg11':
            my_model = models.vgg11(pretrained = True)

        for param in my_model.parameters(): 
            param.requires_grad = False #turning off tuning of the model

        my_model.classifier = load_point['classifier']
        my_model.load_state_dict(load_point['state_dict'])
        my_model.class_to_idx = load_point['mapping']
        my_model.name = load_point['name']
    
    else: # Loading the complete model
        my_model = torch.load(model_path)
    
    my_model.to(device)
    return my_model

def process_image(image):

    pil_image = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    # transform image to tensor
    pil_transform = transform(pil_image)
    
    return pil_transform

def predict(image_path, model, topk=5):
    # processing image
    test_image = process_image(image_path)
    test_image = test_image.to(device)
    test_image = test_image.unsqueeze(dim = 0)
    
    model.eval()
    with torch.no_grad():
        test_prediction = model(test_image)
        test_prediction = torch.exp(test_prediction)
        porbs, classes = test_prediction.topk(topk)
        
    model.train()

    class_list = []
    
    for c in classes[0]:
        for k,v in model.class_to_idx.items():
            if c == v:
                class_list.append(k)
    
    return [round(p,5) for p in porbs[0].tolist()], class_list
    
    
def main():
    #initialize the parser
    parser = argparse.ArgumentParser(description='to get the prediction type: python predict.py <image_path> <model_path> --checkpoint <to load a checkpoint> --top_k <k> --category_names <label_path> --gpu')

    #Add the positional parameters
    parser.add_argument('image', help='Path to the image', type = str)
    parser.add_argument('model', help='Path to load model', type=str)
    #Add the optional parameters
	parser.add_argument('--checkpoint', help='load model checkpoint', default=False, action='store_true')
    parser.add_argument('--top_k', help='Top k predictions', type=int, default=5)
    parser.add_argument('--category_names', help='path to labels map', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', help='Enable gpu', default=False, action='store_true')

    #Parse the argument
    args = parser.parse_args()
    # setting the device
    global device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    #get the model from the model path
    model = load_model(args.model, args.checkpoint)

    # get the labels
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    #predict the image
    prob, classes = predict(args.image, model, args.top_k)

    print('prediction: ', cat_to_name[classes[0]])
    print('top {} probabilities: {}'.format(args.top_k, prob))
    print('top {} classes: {}'.format(args.top_k, classes))

if __name__ == '__main__':
    main()