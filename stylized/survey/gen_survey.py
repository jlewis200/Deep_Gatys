#!/usr/bin/python3

from PIL import Image, ImageDraw, ImageFont
from os import scandir, system
from random import choice

FONT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 80)


def gen_survey_image(path):
    labels = ['A', 'B', 'C', 'D']
    label_map = {}

    for entry in scandir(path):

        if "content" in entry.path:
            label_map["Content"] = entry.path

        elif "style" in entry.path:
            label_map["Style"] = entry.path

        else:
            label = choice(labels)
            labels.remove(label)
            label_map[label] = entry.path

    #store image/label map
    print(label_map)
    system("echo '%s' >> randomization_key.txt" % str(label_map))

    #open the images
    for key in label_map.keys():
        label_map[key] = Image.open(label_map[key])

    #resize the images
    size = label_map["Content"].size
    for key in label_map.keys():
        label_map[key] = label_map[key].resize(size)

    #label the images
    for key in label_map.keys():
        label_img(label_map[key], key)

    #construct the montage
    survey_img = Image.new('RGB', (size[0] * 2, size[1] * 3))

    survey_img.paste(label_map["Content"], (0,       0))
    survey_img.paste(label_map["Style"],   (size[0], 0))
    survey_img.paste(label_map["A"],       (0,       size[1]))
    survey_img.paste(label_map["B"],       (size[0], size[1]))
    survey_img.paste(label_map["C"],       (0,       size[1] * 2))
    survey_img.paste(label_map["D"],       (size[0], size[1] * 2))

    survey_img.save("survey_%s.jpg" % path)


def label_img(img, label):
    ImageDraw.Draw(img).text((10, 10), label, font=FONT, stroke_width=5, fill=(0,0,0), stroke_fill=(255, 255, 255))


system("echo '' > randomization_key.txt")
for entry in scandir():
    if entry.is_dir():
        gen_survey_image(entry.name)




