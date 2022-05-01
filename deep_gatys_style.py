#!/usr/bin/python3

"""
"""

import torch
import random

from torchvision import models, transforms
from torch.nn.functional import interpolate
from torchvision.transforms.functional import rotate 
from PIL import Image
from scipy import ndimage
from time import time
from os import system
from code import interact

#normalization constants from:  https://pytorch.org/vision/stable/models.html
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

#per channel clamping values based on MEAN & STD, shape:  (1, 3, 1, 1)
MINS = (-MEAN / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
MAXS = ((1 - MEAN) / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

def main():
    manual_mode()

def manual_mode():
    ####################################################################################################################
    #  *****update model name when swapping models to ensure image meta data is acccurate*****
    model = models.vgg19(pretrained=True).to("cuda")
    model.name="torchvision.models.vgg19"
    ####################################################################################################################
    print(model)
   
    #replace maxpooling layers with avgpool as per Gatys et al.
    for idx in range(len(model.features)):
        model.features[idx].requires_grad_(False)
        if isinstance(model.features[idx], torch.nn.modules.pooling.MaxPool2d):
            maxpool = model.features[idx]
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=maxpool.kernel_size,
                                                     stride=maxpool.stride,
                                                     padding=maxpool.padding,
                                                     ceil_mode=maxpool.ceil_mode)

    style_layers = [model.features[idx] for idx in [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]]
    #style_layers = [model.features[idx] for idx in [1, 6, 11, 20, 29]]
    content_layer = model.features[22]
  
    content_filename = "5"
    style_filename = "2"
 
    for rotate in [True, False]:
        transfer_style("content/%s.jpg" % content_filename, 
                       "style/%s.jpg" % style_filename,
                       "stylized/deep_gatys/%s_%s_tuning" % (content_filename, style_filename),
                       model,
                       content_layer,
                       style_layers,
                       alpha=1e-10,
                       beta=1e-99,
                       gamma=0,
                       lr=0.01,
                       n_iters=512,
                       n_octave=2,
                       octave_scale=2,
                       rotate=rotate)

def hyperparameter_sweep():
    ####################################################################################################################
    #  *****update model name when swapping models to ensure image meta data is acccurate*****
    model = models.vgg19(pretrained=True).to("cuda")
    model.name="torchvision.models.vgg19"
    ####################################################################################################################
    print(model)

    style_layers = [model.features[idx] for idx in [1, 6, 11, 20, 29]]
    content_layer = model.features[22]
    
    #replace maxpooling layers with avgpool as per Gatys et al.
    for idx in range(len(model.features)):
        model.features[idx].requires_grad_(False)
        if isinstance(model.features[idx], torch.nn.modules.pooling.MaxPool2d):
            maxpool = model.features[idx]
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=maxpool.kernel_size,
                                                     stride=maxpool.stride,
                                                     padding=maxpool.padding,
                                                     ceil_mode=maxpool.ceil_mode)
    
    for content_filename in [str(idx) for idx in range(1, 13)]:
        for style_filename in [str(idx) for idx in range(1, 10)]:
            for alpha in [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_alpha" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               alpha=alpha)

            for beta in [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_beta" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               beta=beta)

            for gamma in [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_gamma" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               gamma=gamma)

            for n_iters in [5, 10, 20, 40]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_n_iters" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               n_iters=n_iters)

            for n_octave in [1, 2, 3, 4, 5, 6]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_n_octave" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               n_octave=n_octave,
                               n_iters=round(40//n_octave)) #maintain total number of iterations for fair comparison

            for octave_scale in [1.2, 1.4, 1.6]:
                transfer_style("content/%s.jpg" % content_filename, 
                               "style/%s.jpg" % style_filename,
                               "stylized/deep_gatys/%s_%s_octave_scale" % (content_filename, style_filename),
                               model,
                               content_layer,
                               style_layers,
                               octave_scale=octave_scale)

def transfer_style(content_filename,  #content image filename
                   style_filename,    #style image filename
                   out_filename,      #output image filename
                   model,             #the pretrained CNN that will be used to generate the image
                   content_layer,     #the content layer of the model
                   style_layers,      #the style layers of the model
                   alpha=1e-10,       #weight of content loss
                   beta=1e-9,         #weight of tv loss
                   gamma=0,           #weight of clipping loss
                   lr=0.2,            #learning rate
                   n_iters=10,        #number of gradient ascent steps per octave
                   n_octave=2,        #number of times to process downsampled images
                   octave_scale=3,    #scale factor for each octave
                   rotate=True):      #apply rotation regulatization

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
    for octave_idx, (content_octave, style_octave) in enumerate(zip(content_octaves[::-1], style_octaves[::-1])):

        #get activations
        content_activation_list = list()
        style_grams_list = list()

        for idx in range(4):
            content_octave = content_octave.rot90(k=1, dims=(2, 3))
            style_octave = style_octave.rot90(k=-1, dims=(2, 3))

            #get content layer activations for content image
            model.forward(content_octave)
            content_activation_list.append(activations[content_layer])

            #get style layer gram matrices for style image
            model.forward(style_octave)
            style_grams_list.append([gram_matrix(activations[style_layer]) for style_layer in style_layers])

        #default activation/grams are original orientation after 4 rotations
        content_activation = content_activation_list[3]
        style_grams = style_grams_list[3]    

        #upsample lower-res detail of previous iteration to shape of current octave
        detail = interpolate(detail, size=content_octave.shape[-2:], align_corners=True, mode="bilinear")        
 
        #add previously accrued detail to the (possibly downsampled) base img
        img = (content_octave.clone().detach() + detail.clone().detach()).detach().requires_grad_(True).cuda()
        optimizer = torch.optim.Adam([img], betas=(0.0, 0.0), weight_decay=0, lr=lr)

        #perform iterative gradient descent for current content_octave
        for idx in range(n_iters):
            if rotate:
                content_activation = content_activation_list[idx % 4]
                style_grams = style_grams_list[idx % 4]

                img = img.rot90(k=1, dims=(2, 3)).clone().detach().requires_grad_(True).cuda()
            
            optimizer = torch.optim.Adam([img], betas=(0.0, 0.0), weight_decay=0, lr=lr)

               
            #backprop to the image
            model.zero_grad()
            optimizer.zero_grad()

            #make forward pass and set objective function
            model.forward(img)

            style_activations = [activations[style_layer] for style_layer in style_layers]
            l_style    = style_loss(style_activations, style_grams)
            l_content  = alpha * content_loss(activations[content_layer], content_activation)
            l_tv       = beta * tv_loss(img)
            l_clipping = gamma * clipping_loss(img)
            loss = l_content + l_style + l_tv + l_clipping

            loss.backward()
            optimizer.step()
       
            print("iter:  %d, %f %f %f %f" %(idx, l_content, l_style, l_tv, l_clipping))
            
        #orient img vertically
        if rotate:
            img = img.rot90(k=(4 - (n_iters % 4)), dims=(2, 3)).cuda()

        #separate accrued detail from the (possibly downsampled) base image
        detail = img - content_octave

        #save the image 
        if octave_idx + 1 == n_octave:
            filename = "%s_%s.jpg" % (out_filename, str(int(time())))
            deprocess(img.clone().cpu()).save(filename)
            hyperparameters = "deep_gatys_style, content_filename=%s, style_filename=%s, model=%s, alpha=%s, beta=%s, gamma=%s, lr=%s, n_iters=%d, n_octave=%d, octave_scale=%s, iter=%d" % (content_filename, style_filename, model.name, str(alpha), str(beta), str(gamma), lr, n_iters, n_octave, octave_scale, idx)
            system("exiftool -overwrite_original -ImageDescription='%s' '%s'" % (hyperparameters, filename)) 

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

def clipping_loss(img):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()
    
    loss = loss + img[img > MAXS].square().sum()
    loss = loss + img[img < MINS].square().sum()

    return loss

def tv_loss(generated_activations):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True).cuda()

    #form a flattened vector containing all rows except final and from that subtract a flattened vector containing all rows except first (across all channels)
    loss = loss + (generated_activations[0, :, 1:].flatten() - generated_activations[0, :, :-1].flatten()).square().sum()

    #form a flattened vector containing all cols except final and from that subtract a flattened vector containing all cols except first (across all channels)
    loss = loss + (generated_activations[0, :, :, 1:].flatten() - generated_activations[0, :, :, :-1].flatten()).square().sum()
    return loss    

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
