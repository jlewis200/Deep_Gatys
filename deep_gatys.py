#!/usr/bin/python3

"""
Nueral style transfer is a method where the style from
one image is applied to the content of another image. This
has applications in artistic settings as well as data augmentation.
"""

import torch

from torchvision import models, transforms
from torch.nn.functional import interpolate
from torchvision.transforms.functional import rotate 
from PIL import Image
from argparse import ArgumentParser

IMAGE_DESCRIPTION_EXIF_TAG = 0x10e

#normalization constants from:  https://pytorch.org/vision/stable/models.html
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

#per channel clamping values based on MEAN & STD, shape:  (1, 3, 1, 1)
MINS = (-MEAN / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
MAXS = ((1 - MEAN) / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

if torch.cuda.is_available():
    MINS = MINS.cuda()
    MAXS = MAXS.cuda()


def main():
    #get command line args
    parser = ArgumentParser(description="Transfer style from a source image to a content image")

    #positional arguments
    parser.add_argument("content", type=str, help="content image path")
    parser.add_argument("style",   type=str, help="style image path")
    parser.add_argument("out",     type=str, help="output image path")

    #options
    parser.add_argument("-a", "--alpha",         type=float, default=1e-10, help="weight of content loss")
    parser.add_argument("-b", "--beta",          type=float, default=1e-99, help="weight of total variational loss")
    parser.add_argument("-g", "--gamma",         type=float, default=0,     help="weight of clipping loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.01,  help="learning rate of optimizer")
    parser.add_argument("-i", "--iterations",    type=int,   default=512,   help="number of optimizer steps per octave")
    parser.add_argument("-k", "--octaves",       type=int,   default=2,     help="number of octaves")
    parser.add_argument("-v", "--octave-scale",  type=float, default=2,     help="downsampling scale between octaves")
    parser.add_argument("-r", "--rotate",        action="store_true",       help="apply rotation regularization")

    args = parser.parse_args()

    #load pre-trained model
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    if torch.cuda.is_available():
        model = model.cuda()

    #replace maxpooling layers with avgpool as per Gatys et al.
    for idx in range(len(model.features)):

        if isinstance(model.features[idx], torch.nn.modules.pooling.MaxPool2d):
            maxpool = model.features[idx]
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=maxpool.kernel_size,
                                                     stride=maxpool.stride,
                                                     padding=maxpool.padding,
                                                     ceil_mode=maxpool.ceil_mode)

    #disable gradients for the model:  the image is trained, the model is not
    model.requires_grad_(False)

    style_layers = [model.features[idx] for idx in [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]]
    content_layer = model.features[22]
  
    transfer_style(args.content, 
                   args.style,
                   args.out,
                   model,
                   content_layer,
                   style_layers,
                   alpha=args.alpha,
                   beta=args.beta,
                   gamma=args.gamma,
                   lr=args.learning_rate,
                   n_iters=args.iterations,
                   n_octave=args.octaves,
                   octave_scale=args.octave_scale,
                   rotate=args.rotate)


def transfer_style(content_filename,  #content image filename
                   style_filename,    #style image filename
                   out_filename,      #output image filename
                   model,             #the pretrained CNN that will be used to generate the image
                   content_layer,     #the content layer of the model
                   style_layers,      #the style layers of the model
                   alpha=1e-10,       #weight of content loss
                   beta=1e-9,         #weight of tv loss
                   gamma=0,           #weight of clipping loss
                   lr=0.02,           #learning rate
                   n_iters=256,       #number of gradient ascent steps per octave
                   n_octave=2,        #number of times to process downsampled images
                   octave_scale=2,    #scale factor for each octave
                   rotate=True):      #apply rotation regulatization

    #expand tensor has 4 dimensions (N x C x H x W)
    content_img = preprocess(Image.open(content_filename))
    style_img = preprocess(Image.open(style_filename))

    if torch.cuda.is_available():
        content_img = content_img.cuda()
        style_img = style_img.cuda()
    
    #down/up sample style image to match content image
    style_img = interpolate(style_img, size=content_img.shape[-2:], align_corners=True, mode="bilinear")
 
    if torch.cuda.is_available():
        style_img = style_img.cuda()
   
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
    detail = torch.zeros_like(content_octaves[-1])

    if torch.cuda.is_available():
        detail = detail.cuda()

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
        img = (content_octave.clone().detach() + detail.clone().detach()).detach().requires_grad_(True)

        if torch.cuda.is_available():
            img = img.cuda()
        optimizer = torch.optim.Adam([img], betas=(0.0, 0.0), weight_decay=0, lr=lr)

        print("iter   content  style    tv       clipping")

        #perform iterative gradient descent for current content_octave
        for idx in range(n_iters):
            if rotate:
                content_activation = content_activation_list[idx % 4]
                style_grams = style_grams_list[idx % 4]

                img = img.rot90(k=1, dims=(2, 3)).clone().detach().requires_grad_(True)
                
                if torch.cuda.is_available():
                    img = img.cuda()
            
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

            print(f"{idx:<5}  {float(l_content):.5f}  {float(l_style):.5f}  {float(l_tv):.5f}  {float(l_clipping):.5f}")
            
        #orient img vertically
        if rotate:
            img = img.rot90(k=(4 - (n_iters % 4)), dims=(2, 3))
            
            if torch.cuda.is_available():
                img = img.cuda()

        #separate accrued detail from the (possibly downsampled) base image
        detail = img - content_octave

        #save the image 
        if octave_idx + 1 == n_octave:
            pil_img = deprocess(img.clone().cpu())

            #set hyperparameters as exif metadata
            exif = pil_img.getexif()
            exif[IMAGE_DESCRIPTION_EXIF_TAG] = f"deep_gatys, " \
                                                "content_filename={content_filename}, " \
                                                "style_filename={style_filename}, " \
                                                "alpha={alpha}, " \
                                                "beta={beta}, " \
                                                "gamma={gamma}, " \
                                                "lr={lr}, " \
                                                "n_iters={n_iters}, " \
                                                "n_octave={n_octave}, " \
                                                "octave_scale={octave_scale}, " \
                                                "iter={idx}"
            pil_img.save(out_filename, exif=exif)


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
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    
    if torch.cuda.is_available():
        loss = loss.cuda()
    
    loss = loss + img[img > MAXS].square().sum()
    loss = loss + img[img < MINS].square().sum()

    return loss


def tv_loss(generated_activations):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    if torch.cuda.is_available():
        loss = loss.cuda()

    #form a flattened vector containing all rows except final and from that subtract a flattened vector containing all rows except first (across all channels)
    loss = loss + (generated_activations[0, :, 1:].flatten() - generated_activations[0, :, :-1].flatten()).square().sum()

    #form a flattened vector containing all cols except final and from that subtract a flattened vector containing all cols except first (across all channels)
    loss = loss + (generated_activations[0, :, :, 1:].flatten() - generated_activations[0, :, :, :-1].flatten()).square().sum()
    return loss    


def style_loss(generated_activations, style_grams):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    if torch.cuda.is_available():
        loss = loss.cuda()

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
