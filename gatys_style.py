#!/usr/bin/python3

"""
Requires pytorch 1.9.0+.  This is the first version of pytorch where the torch.clamp() function accepts tensors instead
of scalars for the min/max values.  This enables per-RGB-channel min/max clamping which helps produce higher-quality
non-saturated images.  The use of per-channel min/max is desirable because the images are normalized using distinct 
mean and standard deviation per-channel.  This result is each channel has different mins/maxs.
"""

import torch
import random

from torchvision import models, transforms
from torch.nn.functional import interpolate
from PIL import Image
from code import interact

#normalization constants from:  https://pytorch.org/vision/stable/models.html
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

def main():
    model = models.vgg19(pretrained=True).to("cuda")
    print(model)
    """
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace=True)
        (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): ReLU(inplace=True)
        (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): ReLU(inplace=True)
        (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): ReLU(inplace=True)
        (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (33): ReLU(inplace=True)
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): ReLU(inplace=True)
        (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
    """

    style_layers = [model.features[idx] for idx in [1, 6, 11, 20, 29]]
    content_layer = model.features[22]
    
    #replace maxpooling layers with avgpool as per Gatys et al.
    for idx in range(len(model.features)):
    
        #interact(local=locals())
        if isinstance(model.features[idx], torch.nn.modules.pooling.MaxPool2d):
            maxpool = model.features[idx]
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=maxpool.kernel_size,
                                                     stride=maxpool.stride,
                                                     padding=maxpool.padding,
                                                     ceil_mode=maxpool.ceil_mode)

    print(model)
    transfer_style("content_images/tuebingen.jpg", 
                   "style_images/starry_night.jpg",
                   "stylized_images/tuebingen_starry_night",
                   model,
                   content_layer,
                   style_layers,
                   10**-11,
                   lr=0.1,
                   n_iters=1001)


def transfer_style(content_filename,  #content image filename
                   style_filename,    #style image filename
                   out_filename,      #output image filename
                   model,             #the pretrained CNN that will be used to generate the image
                   content_layer,     #the content layer of the model
                   style_layers,      #the style layers of the model
                   alpha,             #relative weight of content loss.  style loss weight = 1 - alphaa
                   lr=0.1,            #learning rate
                   n_iters=1000):     #number of gradient ascent steps per octave
    #expand tensor has 4 dimensions (N x C x H x W)
    content_img = preprocess(Image.open(content_filename)).unsqueeze(0).cuda()
    style_img = preprocess(Image.open(style_filename)).unsqueeze(0).cuda()

    #up/down sample style image to match content image
    style_img = interpolate(style_img, size=content_img.shape[-2:], align_corners=True, mode="bilinear").cuda()        
   
    #generate randomly initialized base image
    #passing the random noise through the preprocessing function is important for good results
    #the normalization prevents several failure modes
    generated_img = preprocess(transforms.ToPILImage()(torch.rand(content_img.shape[1:]))).unsqueeze(0).cuda()
    #generated_img = 0.9 * generated_img + 0.1 * content_img
    generated_img = content_img

    #per channel clamping values, shape:  (1, 3, 1, 1)
    mins = (-MEAN / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    maxs = ((1 - MEAN) / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

    #register forward hook to extract layer activations
    #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
   
    content_layer.register_forward_hook(get_activation(content_layer))

    for layer in style_layers:
        layer.register_forward_hook(get_activation(layer))

    
    #place model in eval mode
    model.eval()

    #get content layer activations for content image
    model.forward(content_img)
    content_activation = activations[content_layer]

    #get style layer gram matrices for style image
    model.forward(style_img)
    style_grams = [gram_matrix(activations[style_layer]) for style_layer in style_layers]
   
    #create backprop-able image
    img = generated_img.clone().detach().requires_grad_(True).cuda()
    optimizer = torch.optim.Adam([img], lr=lr)

    #perform iterative gradient ascent for current octave
    for idx in range(n_iters):
        
        if idx in (600, 1000, 1400):
            lr = lr*0.5
        optimizer = torch.optim.Adam([img], lr=lr)

        #backprop to the image
        model.zero_grad()
        optimizer.zero_grad()

        #make forward pass and set objective function
        model.forward(img)

        style_activations = [activations[style_layer] for style_layer in style_layers]
        s_loss = style_loss(style_activations, style_grams)
        c_loss = alpha * content_loss(activations[content_layer], content_activation)
        loss = c_loss + s_loss

        loss.backward()
        optimizer.step()
   
        print("iter:  %d, %f %f" %(idx, c_loss, s_loss))

        #save the image periodically
        if idx  % 100 == 0:
            deprocess(img.clamp(mins, maxs).squeeze(0).clone().cpu()).save("%s_%d.jpg" % (out_filename, idx))

def preprocess(img):
    scale     = transforms.Resize(512)
    convert   = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    transform = transforms.Compose((scale, convert, normalize))
    return transform(img)

def deprocess(img):                                                                                 
    normalize_0 = transforms.Normalize(mean=torch.zeros(3), std=1/STD)
    normalize_1 = transforms.Normalize(mean=-MEAN, std=torch.ones(3))
    convert     = transforms.ToPILImage()
    transform   = transforms.Compose((normalize_0, normalize_1, convert))
    return transform(img)

def style_loss(generated_activations, style_grams):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
    layer_weight = 1.0 / len(generated_activations)  #equal layer weights, summing to 1 as per Gatys et al.
    
    for generated_activation, style_gram in zip(generated_activations, style_grams):
        N_l = generated_activation.shape[1] #number of filters in the layer
        M_l = generated_activation.shape[2] * generated_activation.shape[3] #size of filter
        layer_scaler = layer_weight / (4 * N_l**2 * M_l**2)
        loss = loss + (gram_matrix(generated_activation) - style_gram.detach()).square().sum().mul(layer_scaler)

    return loss

def content_loss(generated_activation, content_activation):
    content_activation = content_activation.detach()
    loss = 0.5 * (generated_activation - content_activation).square().sum()
    return loss

def gram_matrix(activations):
    #flatten dims 2 and 3
    activations = activations.flatten(start_dim=2)

    #transpose flattened activations
    activations_T = activations.permute((0, 2, 1))

    #compute gram matrix for each image via batch matrix multiplication
    gram = activations.bmm(activations_T)

    return gram
   


if __name__ == "__main__":
    main()
