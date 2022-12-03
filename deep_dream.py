#!/usr/bin/python3

#basic use of pretrained model and normalization from:  https://pytorch.org/vision/stable/models.html
import torch
import random
import matplotlib.pyplot as plt
import scipy.ndimage as nd

from torchvision import models, datasets, transforms
from code import interact
from PIL import Image

def main():
    model = models.vgg19(pretrained=True).to("cuda")
    print(model.features)
    
    #deepdream("content_images/clouds.jpeg", 36, model, "clouds_normed", learning_rate=5, num_iterations=200, octave_scale=1.4, n_octave=6)
    #deepdream("content_images/beksinski_1.jpg", 36, model, "beksinski_1", learning_rate=5, num_iterations=100)
    #deepdream("content_images/beksinski_2.jpg", 36, model, "beksinski_2", learning_rate=5, num_iterations=100)
    #deepdream("content_images/beksinski_3.jpg", 36, model, "beksinski_3", learning_rate=5, num_iterations=100, octave_scale=1.2, n_octave=10)
    #deepdream("content_images/beksinski_4.jpg", 36, model, "beksinski_4", learning_rate=5, num_iterations=120, octave_scale=1.4, n_octave=9)
    #deepdream("content_images/beksinski_5.jpg", 36, model, "beksinski_5", learning_rate=5, num_iterations=50, octave_scale=1.4, n_octave=4)
    #deepdream("content_images/beksinski_6.jpg", 36, model, "beksinski_6", learning_rate=5, num_iterations=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/waterfall.jpg", 36, model, "waterfall", learning_rate=5, num_iterations=50, octave_scale=1.4, n_octave=10)
    #deepdream("content_images/4k_skull.jpg", 36, model, "4k_skull", learning_rate=5, num_iterations=100, octave_scale=1.1, n_octave=16)
    deepdream("content_images/fal.jpg", 36, model, "fal", learning_rate=15, num_iterations=101, octave_scale=1.4, n_octave=4)
    exit()

    for target_layer in range(len(model.features)):
        if isinstance(model.features[target_layer], (torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.activation.ReLU)):
            print("maximizing activations for layer %d:  %s\n" % (target_layer, str(model.features[target_layer])))
            deepdream("content_images/clouds.jpeg", target_layer, model, "clouds")

def rescale(x):       
    """
    From a3 image_utils.py
    """

    low, high = x.min(), x.max()                                                                                         
    x_rescaled = (x - low) / (high - low)                                                                                
    return x_rescaled 

def deprocess(img, should_rescale=True):                                                                                 
    """
    From a3 image_utils.py
    """

    lambda_0    = transforms.Lambda(lambda x: x[0])
    normalize_0 = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
    normalize_1 = transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0])
    rescale_0   = transforms.Lambda(rescale)
    convert     = transforms.ToPILImage()

    transform = transforms.Compose((lambda_0, normalize_0, normalize_1, rescale, convert))
    return transform(img)


def deepdream(base_img_filename, 
              target_layer, 
              model, 
              dst_img_filename, 
              learning_rate=1.5,
              num_iterations=10,
              max_jitter=32,
              n_octave=4,
              octave_scale=1.4,
              show_every=100,
              generate_plots=True,
              img_size=(500, 500)):
    """
    Partially based on a3 class_visualizer.py and google deepdream code:  https://github.com/google/deepdream/blob/master/dream.ipynb
    
    Perform gradient ascent on an image to maximize the activations of a target layer.

    Inputs:
    - target_layer:  integer index of the target_layer activations to maximize
    - model: A pretrained CNN that will be used to generate the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    - generate_plots: to plot images or not (used for testing)
    """

    #open source image file, convert to tensor, normalize according to stats provided for pytorch pretrained models
    convert   = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize    = transforms.Resize(img_size)
    #transform = transforms.Compose((convert, normalize, resize))
    transform = transforms.Compose((convert, normalize))

    #ensure tensor has 4 dimensions (N x C x H x W) and requires gradients
    base_img = transform(Image.open(base_img_filename)).unsqueeze(0)

    img_min = base_img.min()
    img_max = base_img.max()
    blur = transforms.GaussianBlur((9, 9), sigma=(0.2, 0.2))

    #register forward hook to extract layer activations
    #adapted from:  https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
    
    model.features[target_layer].register_forward_hook(get_activation(target_layer))
    
    #place model in eval mode
    model.eval()

    octaves = [torch.tensor(nd.zoom(base_img, (1, 1, (1 / octave_scale)**idx, (1 / octave_scale)**idx))).cuda() for idx in range(n_octave)]

    detail = torch.zeros_like(octaves[-1])

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]

        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = torch.tensor(nd.zoom(detail.cpu(), (1, 1, 1.0*h/h1, 1.0*w/w1), order=1)).cuda()        
    
        img = octave_base + detail

        for t in range(num_iterations):
            #apply jitter
            x_jitter = random.randint(0, max_jitter)
            y_jitter = random.randint(0, max_jitter)
            img = img.roll(shifts=(y_jitter, x_jitter), dims=(2, 3))

            img_var = img.clone().detach().requires_grad_(True)

            model.forward(img_var)
            loss = activations[target_layer].sum()
            #loss = torch.linalg.norm(activations[target_layer])
            #interact(local=locals())
            model.zero_grad()
            loss.backward()

            #normalization
            g = img_var.grad
            g /= torch.linalg.norm(g)
            
            img = img + (learning_rate * g)

            #clamp min and max values
            img = img.clamp(img_min, img_max)


    #        if (t + 1) % 10 == 0:
    #            img = blur(img)

            #de-jitter
            img = img.roll(shifts=(-y_jitter, -x_jitter), dims=(2, 3))

        detail = img - octave_base

        # Periodically show the image
        if generate_plots:
            #if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            if True:
                deprocess(img.clone().cpu()).save("visualization/%s_layer_%d_iter_%d.png" % (dst_img_filename, target_layer, t+1))

if __name__ == "__main__":
    main()
