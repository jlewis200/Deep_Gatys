#!/usr/bin/python3

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
    #model = models.vgg19(pretrained=True).to("cuda")
    model = models.googlenet(pretrained=True).to("cuda")
    print(model)
    
    deepdream("content_images/clouds.jpeg", model.inception4e, model, "clouds_normed", step_size=15, n_iters=50, octave_scale=1.4, n_octave=4)
    #deepdream("content_images/beksinski_1.jpg", 36, model, "beksinski_1", step_size=5, n_iters=100)
    #deepdream("content_images/beksinski_2.jpg", 36, model, "beksinski_2", step_size=5, n_iters=100)
    #deepdream("content_images/beksinski_3.jpg", 36, model, "beksinski_3", step_size=5, n_iters=100, octave_scale=1.2, n_octave=10)
    #deepdream("content_images/beksinski_4.jpg", 36, model, "beksinski_4", step_size=5, n_iters=120, octave_scale=1.4, n_octave=9)
    #deepdream("content_images/beksinski_4.jpg", model.inception4c, model, "beksinski_4", step_size=5, n_iters=10, octave_scale=1.4, n_octave=9)
    #deepdream("content_images/beksinski_5.jpg", 36, model, "beksinski_5", step_size=5, n_iters=50, octave_scale=1.4, n_octave=4)
    #deepdream("content_images/beksinski_6.jpg", 36, model, "beksinski_6", step_size=5, n_iters=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/waterfall.jpg", 36, model, "waterfall", step_size=5, n_iters=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/4k_skull.jpg", 36, model, "4k_skull", step_size=5, n_iters=100, octave_scale=1.1, n_octave=16)
    exit()

    for target_layer in range(len(model.features)):
        if isinstance(model.features[target_layer], (torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.activation.ReLU)):
            print("maximizing activations for layer %d:  %s\n" % (target_layer, str(model.features[target_layer])))
            deepdream("content_images/clouds.jpeg", target_layer, model, "clouds")

def deepdream(in_filename, #input image filename
              target_layer,      #a layer in the model which can have a forward hook registered
              model,             #the pretrained CNN that will be used to generate the image
              out_filename,  #output image filename
              step_size=6.0,     #higher learning rate BC we account for ImageNet STD
              n_iters=10,        #number of gradient ascent steps per octave
              jitter=32,         #max number of pixels to shift in x/y direction
              n_octave=4,        #number of times to process downsampled images
              octave_scale=1.4): #scale factor for each octave
    """
    Based on Google Deep Dream:  https://github.com/google/deepdream/blob/master/dream.ipynb
    
    Perform gradient ascent on an image to maximize the activations of a target layer.

    - n_iters:  Number of iterations per octave
    - jitter:  M 
    """

    #ensure tensor has 4 dimensions (N x C x H x W)
    base_img = preprocess(Image.open(in_filename)).unsqueeze(0).cuda()

    #clamping values, shape:  (1, 3, 1, 1)
    mins = (-MEAN / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    maxs = ((1 - MEAN) / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

    #register forward hook to extract layer activations
    #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
    
    target_layer.register_forward_hook(get_activation(target_layer))
    
    #place model in eval mode
    model.eval()

    #generate zoomed-out/lower-res images, octaves[0]:  original resolution, octaves[-1]:  lowest resolution
    octaves = [interpolate(base_img, scale_factor=1/octave_scale**idx, recompute_scale_factor=False, align_corners=True, mode="bilinear") for idx in range(n_octave)]

    #detail tensor accumulates the changes made to the image during the iterative gradient ascent
    detail = torch.zeros_like(octaves[-1]).cuda()

    #iterate through images from lowest to highest resolution
    for octave in octaves[::-1]:

        #upsample lower-res detail of previous iteration to shape of current octave
        detail = interpolate(detail, size=octave.shape[-2:], align_corners=True, mode="bilinear")        
   
        #add previously accrued detail to the (possibly downsampled) base img
        img = octave + detail

        #perform iterative gradient ascent for current octave
        for idx in range(n_iters):
            
            #apply jitter
            x_jitter = random.randint(0, jitter)
            y_jitter = random.randint(0, jitter)
            img = img.roll(shifts=(y_jitter, x_jitter), dims=(2, 3))

            #create backprop-able image
            img_var = img.clone().detach().requires_grad_(True)

            #make forward pass and set loss function
            model.forward(img_var)
            loss = activations[target_layer].sum()
            #loss = torch.linalg.norm(activations[target_layer])

            #backprop to the image
            model.zero_grad()
            loss.backward()

            #normalize gradients
            g = img_var.grad
            g /= torch.linalg.norm(g) #norm is a deviation, original used abs().mean()
            
            #gradient step
            img = img + (step_size * g)
            
            #clamp min and max values
            img = img.clamp(mins, maxs) #clamping is a deviation, original had none

            #de-jitter
            img = img.roll(shifts=(-y_jitter, -x_jitter), dims=(2, 3))

        #separate accrued detail from the (possibly downsampled) base image
        detail = img - octave

        #save the image on each octave
        deprocess(img.squeeze(0).clone().cpu()).save("visualization/%s_iter_%d.png" % (out_filename, idx))

def preprocess(img):
    convert   = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    transform = transforms.Compose((convert, normalize))
    return transform(img)

def deprocess(img):                                                                                 
    normalize_0 = transforms.Normalize(mean=torch.zeros(3), std=1/STD)
    normalize_1 = transforms.Normalize(mean=-MEAN, std=torch.ones(3))
    convert     = transforms.ToPILImage()
    transform   = transforms.Compose((normalize_0, normalize_1, convert))
    return transform(img)

if __name__ == "__main__":
    main()
