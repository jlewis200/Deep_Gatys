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
    #model = models.vgg19(pretrained=True).to("cuda")
    #model = models.googlenet(pretrained=True).to("cuda")
    model = models.convnext_base(pretrained=True).to("cuda")
    print(model)
    
    #deepdream("content_images/clouds.jpeg", model.inception4e, model, "clouds_normed", step_size=15, n_iters=50, octave_scale=1.4, n_octave=4)
    #deepdream("content_images/beksinski_1.jpg", 36, model, "beksinski_1", step_size=5, n_iters=100)
    #deepdream("content_images/beksinski_2.jpg", 36, model, "beksinski_2", step_size=5, n_iters=100)
    #deepdream("content_images/beksinski_3.jpg", 36, model, "beksinski_3", step_size=5, n_iters=100, octave_scale=1.2, n_octave=10)
    #deepdream("content_images/beksinski_4.jpg", 36, model, "beksinski_4", step_size=5, n_iters=120, octave_scale=1.4, n_octave=9)
    #deepdream("content_images/beksinski_4.jpg", model.inception4c, model, "beksinski_4", step_size=5, n_iters=10, octave_scale=1.4, n_octave=9)
    #deepdream("content_images/beksinski_5.jpg", 36, model, "beksinski_5", step_size=5, n_iters=50, octave_scale=1.4, n_octave=4)
    #deepdream("content_images/beksinski_6.jpg", 36, model, "beksinski_6", step_size=5, n_iters=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/waterfall.jpg", 36, model, "waterfall", step_size=5, n_iters=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/4k_skull.jpg", 36, model, "4k_skull", step_size=5, n_iters=100, octave_scale=1.1, n_octave=16)
    #exit()

    #enumerate convnext layers for style transfer
    for idx in range(7, 0, -1):
        target_layer = model.features[idx]
        #register forward hook to extract layer activations
        #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output
            return hook
        target_layer.register_forward_hook(get_activation(target_layer))
        model.forward(preprocess(Image.open("style_images/flowers.jpeg")).unsqueeze(0).cuda())
        model.eval()
        target_activations = activations[target_layer]

    #    interact(local=locals())
        def style_objective(x):
            x = x.squeeze().flatten(1)
            y = target_activations.clone().detach()#.requires_grad_(True)
            y = y.squeeze().flatten(1)
            a = x.T @ y
            a = y[:, a.argmax(1)]
            a = a / x

            #a = torch.linalg.norm(a)
            #interact(local=locals())
            #a = a.sum() 
            return a
        

        deepdream("content_images/clouds.jpeg", target_layer, style_objective, model, "convnext_base/clouds_layer_%d" % idx, step_size=15, n_iters=100, octave_scale=1.4, n_octave=4)


#    #enumerate VGG layers for style transfer
#    for idx in range(36, 8, -1):
#        target_layer = model.features[idx]
#        #register forward hook to extract layer activations
#        #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
#        activations = {}
#        
#        def get_activation(name):
#            def hook(model, input, output):
#                activations[name] = output
#            return hook
#        target_layer.register_forward_hook(get_activation(target_layer))
#        model.forward(preprocess(Image.open("style_images/flowers.jpeg")).unsqueeze(0).cuda())
#        model.eval()
#        target_activations = activations[target_layer]
#
#    #    interact(local=locals())
#        def style_objective(x):
#            x = x.squeeze().flatten(1)
#            y = target_activations.clone().detach()#.requires_grad_(True)
#            y = y.squeeze().flatten(1)
#            a = x.T @ y
#            a = y[:, a.argmax(1)]
#            a = a / x
#
#            #a = torch.linalg.norm(a)
#            #interact(local=locals())
#            #a = a.sum() 
#            return a
#        
#
#        deepdream("content_images/clouds.jpeg", target_layer, style_objective, model, "clouds_layer_%d" % idx, step_size=15, n_iters=100, octave_scale=1.4, n_octave=4)
    #idx=36
    #deepdream("content_images/mat.jpg", model.fc, lambda x : torch.linalg.norm(x), model, "mat_layer_%d" % idx, step_size=10, n_iters=100, octave_scale=1.7, n_octave=4)
    
#    for idx, target_layer in enumerate(model.features):
#        if isinstance(target_layer, (torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.activation.ReLU)):
#            print("maximizing activations for layer:  %s\n" % str(target_layer))
#            deepdream("content_images/clouds.jpeg", target_layer, sum_objective_function, model, "clouds_layer_%d" % idx, step_size=5, n_iters=200, octave_scale=1.4, n_octave=4)

def deepdream(in_filename,       #input image filename
              target_layer,      #a layer in the model which can have a forward hook registered
              objective_function,#a lambda style objective function to apply to the activations of the target layer
              model,             #the pretrained CNN that will be used to generate the image
              out_filename,      #output image filename
              step_size=6.0,     #higher learning rate BC we account for ImageNet STD
              n_iters=10,        #number of gradient ascent steps per octave
              jitter=32,         #max number of pixels to shift in x/y direction
              n_octave=4,        #number of times to process downsampled images
              octave_scale=1.4): #scale factor for each octave
    """
    Based on Google Deep Dream:  https://github.com/google/deepdream/blob/master/dream.ipynb
    
    Perform gradient ascent on an image to maximize the activations of a target layer after applying a objective function.

    Example objective functions:
    lambda x : x.sum() #sum all activations
    lambda x : lambda x : torch.linalg.norm(x)) #take norm of all activations
    lambda x : x[0, 65] #optimize for a particular class.  Class indexes:  https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    lambda x : x[0, [65, 66]].sum() #optimize for a number of classes
    """

    #expand tensor has 4 dimensions (N x C x H x W)
    base_img = preprocess(Image.open(in_filename)).unsqueeze(0).cuda()

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

            #make forward pass and set objective function
            model.forward(img_var)
            objective = objective_function(activations[target_layer])

            #backprop to the image
            model.zero_grad()
            a = activations[target_layer]
            objective = objective.reshape(a.shape)
            torch.autograd.backward(a, grad_tensors=objective)
            #interact(local=locals())
            #objective.backward()

            #normalize gradients
            g = img_var.grad
            g /= torch.linalg.norm(g) #norm is a deviation, original used abs().mean()
            
            #gradient step
            #interact(local=locals())
            img = img + (step_size * g)
            
            #clamp min and max values
            img = img.clamp(mins, maxs) #clamping is a deviation, original had none

            #de-jitter
            img = img.roll(shifts=(-y_jitter, -x_jitter), dims=(2, 3))

        #separate accrued detail from the (possibly downsampled) base image
        detail = img - octave

        #save the image on each octave
        deprocess(img.squeeze(0).clone().cpu()).save("stylized_images/%s.png" % out_filename)

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
