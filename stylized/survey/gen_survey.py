#!/usr/bin/python3

from PIL import Image, ImageDraw, ImageFont
from os import scandir
from random import choice

font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)


def gen_survey_image(path):
    labels = ['A', 'B', 'C']
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

    for key in label_map.keys():
        label_map[key] = label_img(label_map[key], key)

    size = label_map["Content"].size

    for key in label_map.keys():
        label_map[key] = label_map[key].resize(size)

    survey_img = Image.new('RGB', (size[0] * 2, size[1] * 3))

    survey_img.paste(label_map["Content"], (0,       0))
    survey_img.paste(label_map["Style"],   (size[0], 0))
    survey_img.paste(label_map["A"],       (0,       size[1]))
    survey_img.paste(label_map["B"],       (size[0], size[1]))
    survey_img.paste(label_map["C"],       (0,       size[1] * 2))

    survey_img.save("survey_%s.jpg")


def label_img(path, label):
    img = Image.open(path)
    ImageDraw.Draw(img).text((10, 10), label, font=FONT, fill=(0,0,0))
    return img



for entry in scandir():
    if entry.is_dir:
        gen_survey_image(entry.path)




