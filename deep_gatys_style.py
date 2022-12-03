#!/usr/bin/python3

"""
"""

import torch
import random

from torchvision import models, transforms
from torch.nn.functional import interpolate
from PIL import Image
from os import system
from code import interact

#normalization constants from:  https://pytorch.org/vision/stable/models.html
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

def main():
    ####################################################################################################################
    #  *****update model name when swapping models to ensure image meta data is acccurate*****
    model = models.vgg19(pretrained=True).to("cuda")
    model.name="torchvision.models.vgg19"
    ####################################################################################################################

    style_layers = [model.features[idx] for idx in [1, 6, 11, 20, 29]]
    content_layer = model.features[22]
    
    #replace maxpooling layers with avgpool as per Gatys et al.
    for idx in range(len(model.features)):
        if isinstance(model.features[idx], torch.nn.modules.pooling.MaxPool2d):
            maxpool = model.features[idx]
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=maxpool.kernel_size,
                                                     stride=maxpool.stride,
                                                     padding=maxpool.padding,
                                                     ceil_mode=maxpool.ceil_mode)

    transfer_style("content_images/cat.jpg", 
                   "style_images/painting3.jpg",
                   "stylized_images/cat_bricks_deep",
                   model,
                   content_layer,
                   style_layers,
                   1e-11,
                   lr=0.15,
                   n_iters=100,
                   n_octave=4,
                   detail_decay=1)


def transfer_style(content_filename,  #content image filename
                   style_filename,    #style image filename
                   out_filename,      #output image filename
                   model,             #the pretrained CNN that will be used to generate the image
                   content_layer,     #the content layer of the model
                   style_layers,      #the style layers of the model
                   alpha=1e-11,             #relative weight of content loss.  style loss weight = 1 - alphaa
                   lr=0.1,            #learning rate
                   n_iters=1001,      #number of gradient ascent steps per octave
                   n_octave=4,        #number of times to process downsampled images
                   octave_scale=1.4,  #scale factor for each octave
                   detail_decay=1):   #scaler for previous octave's detail
     

    #expand tensor has 4 dimensions (N x C x H x W)
    content_img = preprocess(Image.open(content_filename)).cuda()
    style_img = preprocess(Image.open(style_filename)).cuda()

    #down/up sample style image to match content image
    style_img = interpolate(style_img, size=content_img.shape[-2:], align_corners=True, mode="bilinear").cuda()        
   
    #register forward hook to extract layer activations
    #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    activations = {}
    
    def get_activation(self, input, output):
        activations[self] = output
   
    content_layer.register_forward_hook(get_activation)

    for layer in style_layers:
        layer.register_forward_hook(get_activation)
    
    #place model in eval mode
    model.eval()

    #generate zoomed-out/lower-res images, content_octaves[0]:  original resolution, content_octaves[-1]:  lowest resolution
    content_octaves = [interpolate(content_img, scale_factor=1/octave_scale**idx, recompute_scale_factor=False, align_corners=True, mode="bilinear") for idx in range(n_octave)]
    style_octaves = [interpolate(style_img, scale_factor=1/octave_scale**idx, recompute_scale_factor=False, align_corners=True, mode="bilinear") for idx in range(n_octave)]

    #detail tensor accumulates the changes made to the image during the iterative gradient ascent
    detail = torch.zeros_like(content_octaves[-1]).cuda()

    #iterate through images from lowest to highest resolution
    for content_octave, style_octave in zip(content_octaves[::-1], style_octaves[::-1]):

        #upsample lower-res detail of previous iteration to shape of current octave
        detail = interpolate(detail, size=content_octave.shape[-2:], align_corners=True, mode="bilinear")        
   
        
        #get content layer activations for content image
        model.forward(content_octave)
        content_activation = activations[content_layer]

        #get style layer gram matrices for style image
        model.forward(style_octave)
        style_grams = [gram_matrix(activations[style_layer]) for style_layer in style_layers]
       
        #add previously accrued detail to the (possibly downsampled) base img
        img = (content_octave.clone().detach() + detail_decay * detail.clone().detach()).detach().requires_grad_(True).cuda()
        optimizer = torch.optim.Adam([img], betas=(0.9, 0.999), weight_decay=0, lr=lr)

        #perform iterative gradient descent for current content_octave
        for idx in range(n_iters):
            
            if idx in (600, 1000, 1400, 2000):
                lr = lr*0.5
                optimizer.lr = lr

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
                deprocess(img.clone().cpu()).save("%s_%d.jpg" % (out_filename, idx))
                hyperparameters = "deep_gatys_style, content_filename=%s, style_filename=%s, model=%s, alpha=%s, lr=%s, n_iters=%d, n_octave=%d, octave_scale=%s, detail_decay=%s, iter=%d" % (content_filename, style_filename, model.name, str(alpha), lr, n_iters, n_octave, octave_scale, detail_decay, idx)
                system("exiftool -ImageDescription='%s' '%s_%d.jpg'" % (hyperparameters, out_filename, idx)) 

        #separate accrued detail from the (possibly downsampled) base image
        detail = img - content_octave

        #save the image on each octave
        deprocess(img.squeeze(0).clone().cpu()).save("%s.jpg" % out_filename)
        hyperparameters = "deep_gatys_style, content_filename=%s, style_filename=%s, model=%s, alpha=%s, lr=%s, n_iters=%d, n_octave=%d, octave_scale=%s, detail_decay=%s, iter=%d" % (content_filename, style_filename, model.name, str(alpha), lr, n_iters, n_octave, octave_scale, detail_decay, idx)
        system("exiftool -ImageDescription='%s' '%s.jpg'" % (hyperparameters, out_filename)) 




def preprocess(img):
    convert   = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    unsqueeze = transforms.Lambda(lambda img : img.unsqueeze(0))
    transform = transforms.Compose((convert, normalize, unsqueeze))
    return transform(img)

def deprocess(img):                                                                                 
    squeeze     = transforms.Lambda(lambda img : img.squeeze(0))
    normalize_0 = transforms.Normalize(mean=torch.zeros(3), std=1/STD)
    normalize_1 = transforms.Normalize(mean=-MEAN, std=torch.ones(3))
    clamp       = transforms.Lambda(lambda img : img.clamp(0, 1))
    convert     = transforms.ToPILImage()
    transform   = transforms.Compose((squeeze, normalize_0, normalize_1, clamp, convert))
    return transform(img)

def style_loss(generated_activations, style_grams):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
    layer_weight = 1.0 / len(generated_activations)  #equal layer weights, summing to 1 as per Gatys et al.
    
    for generated_activation, style_gram in zip(generated_activations, style_grams):
        layer_scaler = layer_weight / (4 * generated_activation.numel()**2)
        loss = loss + (gram_matrix(generated_activation) - style_gram.detach()).square().sum().mul(layer_scaler)

    return loss

def content_loss(generated_activation, content_activation):
    loss = (generated_activation - content_activation.detach()).square().sum()
    return loss

def gram_matrix(activations):
    #flatten dims 2 and 3
    activations = activations.flatten(start_dim=2)

    #compute gram matrix for each image via batch matrix multiplication
    return activations.bmm(activations.permute((0, 2, 1)))

if __name__ == "__main__":
    main()
