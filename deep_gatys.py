#!/usr/bin/python3

"""
Nueral style transfer is a method where the style from
one image is applied to the content of another image. This
has applications in artistic settings as well as data augmentation.
"""

from argparse import ArgumentParser

# pylint: disable=no-member
import torch

from torchvision import models, transforms
from torch.nn.functional import interpolate
from PIL import Image

IMAGE_DESCRIPTION_EXIF_TAG = 0x10e

# normalization constants from:  https://pytorch.org/vision/stable/models.html
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def main():
    """
    Perform a single style transfer using the specified by command line args.
    """

    # get command line args
    parser = ArgumentParser(description="Transfer style from a source image to a content image")

    # positional arguments
    parser.add_argument("content", type=str, help="content image path")
    parser.add_argument("style",   type=str, help="style image path")
    parser.add_argument("out",     type=str, help="output image path")

    # options
    parser.add_argument("-a", "--alpha", type=float, default=1e-10, help="weight of content loss")
    parser.add_argument("-b", "--beta", type=float, default=1e-99, help="weight of total variational loss")
    parser.add_argument("-g", "--gamma", type=float, default=0, help="weight of clipping loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.01, help="learning rate of optimizer")
    parser.add_argument("-i", "--iterations", type=int, default=512, help="number of optimizer steps per octave")
    parser.add_argument("-k", "--octaves", type=int, default=2, help="number of octaves")
    parser.add_argument("-v", "--octave-scale", type=float, default=2, help="downsampling scale between octaves")
    parser.add_argument("-r", "--rotate", action="store_true", help="apply rotation regularization")

    args = parser.parse_args()

    # load pre-trained model, place in eval mode
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    if torch.cuda.is_available():
        model = model.cuda()

    # replace maxpooling layers with avgpool as per Gatys et al.
    for idx, feature in enumerate(model.features):
        if isinstance(feature, torch.nn.modules.pooling.MaxPool2d):
            model.features[idx] = torch.nn.AvgPool2d(kernel_size=feature.kernel_size,
                                                     stride=feature.stride,
                                                     padding=feature.padding,
                                                     ceil_mode=feature.ceil_mode)

    # place in eval mode and disable model gradients:  the image is trained, not the model
    model.eval()
    model.requires_grad_(False)

    # get the style and content layers from the model
    style_layer_indices = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]
    model.style_layers = [model.features[idx] for idx in style_layer_indices]
    model.content_layer = model.features[22]
    register_activations(model)

    transfer_style(args.content,
                   args.style,
                   args.out,
                   model,
                   alpha=args.alpha,
                   beta=args.beta,
                   gamma=args.gamma,
                   learning_rate=args.learning_rate,
                   n_iters=args.iterations,
                   n_octave=args.octaves,
                   octave_scale=args.octave_scale,
                   rotate=args.rotate)


def transfer_style(content_filename,    # content image filename
                   style_filename,      # style image filename
                   out_filename,        # output image filename
                   model,               # the pretrained CNN that will be used to generate the image
                   alpha=1e-10,         # weight of content loss
                   beta=1e-9,           # weight of tv loss
                   gamma=0,             # weight of clipping loss
                   learning_rate=0.02,  # learning rate
                   n_iters=256,         # number of gradient ascent steps per octave
                   n_octave=2,          # number of times to process downsampled images
                   octave_scale=2,      # scale factor for each octave
                   rotate=True):        # apply rotation regulatization
    """
    Perform style transfer.
    """

    # preprocessed tensor has 4 dimensions (N x C x H x W)
    content_img = preprocess(Image.open(content_filename))
    style_img = preprocess(Image.open(style_filename))

    # down/up sample style image to match content image
    style_img = interpolate(style_img,
                            size=content_img.shape[-2:],
                            align_corners=True,
                            mode="bilinear")

    # downsample to generate zoomed-out/lower-res content style images
    # content_octaves[0]:  original resolution; content_octaves[-1]:  lowest resolution
    content_octaves = [interpolate(content_img,
                                   scale_factor=1/octave_scale**idx,
                                   recompute_scale_factor=False,
                                   align_corners=True,
                                   mode="bilinear") for idx in range(n_octave)]

    style_octaves = [interpolate(style_img,
                                 scale_factor=1/octave_scale**idx,
                                 recompute_scale_factor=False,
                                 align_corners=True,
                                 mode="bilinear") for idx in range(n_octave)]

    # detail tensor accumulates the changes made to the image during the iterative gradient ascent
    detail = torch.zeros_like(content_octaves[-1])

    # iterate through images from lowest to highest resolution
    for content_octave, style_octave in zip(content_octaves[::-1], style_octaves[::-1]):

        # get 4 rotated content layer activations for the content image
        content_activations = get_rotated_content_activations(model, content_octave, rotate)

        # get 4 rotated style layer activations for the style image
        # there are multiple style layers so this is a list of lists of tensors
        style_activations = get_rotated_style_activations(model, style_octave, rotate)

        # upsample lower-res detail of previous iteration to shape of current octave
        detail = interpolate(detail,
                             size=content_octave.shape[-2:],
                             align_corners=True,
                             mode="bilinear")

        # add previously accrued detail to the (possibly downsampled) base img
        img = content_octave + detail

        print("iter   content  style    tv       clipping")

        # perform optimization steps for current content/style octaves
        for idx in range(n_iters):
            # get the rotated content and style layer activations
            content_activation = content_activations[idx % 4]
            style_activation = style_activations[idx % 4]

            # rotate the image 90 degrees;  detach to make img a leaf tensor
            img = img.rot90(k=int(rotate), dims=(2, 3)).detach().requires_grad_(True)

            # initialize a new optimizer for the rotated image
            optimizer = torch.optim.Adam([img], lr=learning_rate)

            # clear model gradients
            model.zero_grad()

            # make forward pass
            model.forward(img)

            # get the content layer activation/grams for the generated image
            img_content_activation = get_content_activation(model)
            img_style_activation = get_style_activation(model)

            # calculate losses
            l_style = style_loss(img_style_activation, style_activation)
            l_content = alpha * content_loss(img_content_activation, content_activation)
            l_tv = beta * tv_loss(img)
            l_clipping = gamma * clipping_loss(img)
            loss = l_content + l_style + l_tv + l_clipping

            # backprop to the image and take step to optimize image wrt losses
            loss.backward()
            optimizer.step()

            # pytorch tensors don't have proper f-string handlers
            # so losses are explicitly converted using float()
            print(f"{idx:<5}  "
                  f"{float(l_content):.5f}  "
                  f"{float(l_style):.5f}  "
                  f"{float(l_tv):.5f}  "
                  f"{float(l_clipping):.5f}")

        # unrotate img back to vertical
        img = img.rot90(k=-n_iters * int(rotate), dims=(2, 3))

        # separate accrued detail from the (possibly downsampled) base image
        detail = img - content_octave

    # deprocess img to obtain PIL Image
    pil_img = deprocess(img)

    # set hyperparameters as exif metadata
    exif = pil_img.getexif()
    exif[IMAGE_DESCRIPTION_EXIF_TAG] = f"content_filename={content_filename}, " \
                                       f"style_filename={style_filename}, " \
                                       f"alpha={alpha}, " \
                                       f"beta={beta}, " \
                                       f"gamma={gamma}, " \
                                       f"learning_rate={learning_rate}, " \
                                       f"n_iters={n_iters}, " \
                                       f"n_octave={n_octave}, " \
                                       f"octave_scale={octave_scale}, " \
                                       f"iter={idx}"

    # save the image
    pil_img.save(out_filename, exif=exif)


def get_content_activation(model):
    """
    Get the activations of the content layer for the previous forward pass of the model.
    """

    return model.activations[model.content_layer]


def get_style_activation(model):
    """
    Get the activations of the style layers for the previous forward pass of the model.
    """

    return [model.activations[style_layer] for style_layer in model.style_layers]


def get_rotated_content_activations(model, content_octave, rotate):
    """
    Get content layer activations across the four rotations.
    """

    content_activation_list = []

    for _ in range(4):
        content_octave = content_octave.rot90(k=int(rotate), dims=(2, 3))
        model.forward(content_octave)
        content_activation_list.append(get_content_activation(model))

    return content_activation_list


def get_rotated_style_activations(model, style_octave, rotate):
    """
    Get gram matrices from style image across the four rotations.
    """

    style_grams_list = []

    for _ in range(4):
        style_octave = style_octave.rot90(k=int(rotate), dims=(2, 3))
        model.forward(style_octave)
        style_grams_list.append(get_style_activation(model))

    return style_grams_list


def register_activations(model):
    """
    Use the register forward hook to extract layer activations.  Adapted from:
    https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301
    """

    model.activations = {}

    def get_activation(self, _, out):
        model.activations[self] = out

    model.content_layer.register_forward_hook(get_activation)

    for layer in model.style_layers:
        layer.register_forward_hook(get_activation)


def preprocess(img):
    """
    Accept a PIL image, normalize, return a tensor of the appropriate shape.
    """

    # the model was trained with an RGB color model; convert non-RGB (RGBA, CMYK, etc.)
    img = img.convert("RGB")

    convert = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    unsqueeze = transforms.Lambda(lambda img: img.unsqueeze(0))
    transform = transforms.Compose((convert, normalize, unsqueeze))

    img = transform(img)

    if torch.cuda.is_available():
        img = img.cuda()

    return img


def deprocess(img):
    """
    Accept an image as a tensor, invert the preprocess normalization, return a
    PIL image.
    """

    squeeze = transforms.Lambda(lambda img: img.squeeze(0))
    normalize_0 = transforms.Normalize(mean=torch.zeros(3), std=1/STD)
    normalize_1 = transforms.Normalize(mean=-MEAN, std=torch.ones(3))
    clamp = transforms.Lambda(lambda img: img.clamp(0, 1))
    convert = transforms.ToPILImage()
    transform = transforms.Compose((squeeze, normalize_0, normalize_1, clamp, convert))

    return transform(img)


def clipping_loss(img):
    """
    This loss function penalizes pixels which stray outside of the proper RGB
    range.
    """

    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # per channel clamping values based on MEAN & STD, shape:  (1, 3, 1, 1)
    mins = (-MEAN / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    maxs = ((1 - MEAN) / STD).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    if torch.cuda.is_available():
        loss = loss.cuda()
        mins = mins.cuda()
        maxs = maxs.cuda()

    loss = loss + img[img > maxs].square().sum()
    loss = loss + img[img < mins].square().sum()

    return loss


def tv_loss(activations):
    """
    This loss function penalizes adjacent pixels with very large differences.
    """

    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    if torch.cuda.is_available():
        loss = loss.cuda()

    # form a flattened vector containing all rows except final and from that subtract
    # a flattened vector containing all rows except first (across all channels)
    loss = loss + \
           (activations[0, :, 1:].flatten() - activations[0, :, :-1].flatten()).square().sum()

    # form a flattened vector containing all cols except final and from that subtract
    # a flattened vector containing all cols except first (across all channels)
    loss = loss + \
           (activations[0, :, :, 1:].flatten() - activations[0, :, :, :-1].flatten()).square().sum()

    return loss


def style_loss(generated_activations, style_activations):
    """
    This loss function penalizes style differences between the generated image
    and the style image.
    """

    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    if torch.cuda.is_available():
        loss = loss.cuda()

    # equal layer weights, summing to 1 as per Gatys et al.
    layer_weight = 1.0 / len(generated_activations)

    for generated_activation, style_activation in zip(generated_activations, style_activations):
        layer_scaler = layer_weight / (4 * generated_activation.numel()**2)
        gram_diff = gram_matrix(generated_activation) - gram_matrix(style_activation)
        loss = loss + gram_diff.square().sum().mul(layer_scaler)

    return loss


def content_loss(generated_activation, content_activation):
    """
    This loss function penalizes content differences between the generated image
    and the content image.
    """

    return (generated_activation - content_activation).square().sum()


def gram_matrix(activations):
    """
    Returns the gram matrix given a set of layer activations.
    """

    # flatten dims 2 and 3
    activations = activations.flatten(start_dim=2)

    # compute gram matrix for each image via batch matrix multiplication
    return activations.bmm(activations.permute((0, 2, 1)))


if __name__ == "__main__":
    main()
