import os
import sys

import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms


def inint_model(pretrained_model="./output/FCN_ResNet101.pth"):
    model = torchvision.models.segmentation.__dict__['fcn_resnet101'](num_classes=4,
                                                                      aux_loss=False,
                                                                      pretrained=False)

    state_dict = torch.load(pretrained_model)
    model.load_state_dict(state_dict['model'])

    model.eval()

    return model


def infer(model, imgpath):
    input_image = Image.open(imgpath)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    palette = torch.tensor([0, 0, 128])
    colors = torch.as_tensor([i for i in range(4)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    r.save(imgpath.replace('.jpg', '.png'))


if __name__ == '__main__':
    model = inint_model()
    for jpg in os.listdir('./imgs'):
        infer(model, os.path.join('./imgs', jpg))
