from typing import Callable
from matplotlib import pyplot as plt
import numpy as np

from pytorch_lightning import LightningModule

import torch
from torch import Tensor
from torchvision.transforms.functional import rotate


def show_tensor_image(tensor_image: Tensor) -> None:
    """
    Displays tensor as image
    """
    plt.imshow(tensor_image[0])
    plt.axis('off')


def rotating_image_classification(model: LightningModule, image: Tensor):
    """
    Displays graphs with the effect rotating an image
    has on classification, similar to the graphs shown
    in the original paper
    """
    classification_threshold = 0.5
    max_rotation_angle = 180
    rotation_steps = 10
    single_rotation_angle = max_rotation_angle / rotation_steps

    height, width = image.shape[2:]
    image_display = np.zeros((height, width * rotation_steps))
    image_display.shape
    all_probabilities = []
    uncertainties = []
    angles = []

    for step in range(rotation_steps):
        step_rotation_angle = single_rotation_angle * step
        angles.append(step_rotation_angle)
        rotated_image = rotate(image, step_rotation_angle)
        output = model.predict(rotated_image)
        if len(output) > 2:
            _, probabilities, uncertainty = output
            uncertainties.append(int(uncertainty))
        else:
            _, probabilities = output
        all_probabilities.append(probabilities.view(-1,1))
        image_display[0:height, step * width:(step+1)*width] = rotated_image

    display_probabilities = torch.cat(all_probabilities, dim=1)

    # probabilities plot:
    plt.figure(figsize=(20,10))
    for class_no, class_probabilities in enumerate(display_probabilities):
        class_probabilities = class_probabilities.tolist()
        if max(class_probabilities) < classification_threshold:
            continue
        plt.plot(angles, class_probabilities, label=f'{class_no}')
    plt.plot(angles, uncertainties, label='uncertainty', c='red', linestyle='dotted')
    plt.legend()
    plt.xlim([0, max_rotation_angle])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')

    # rotated images:
    plt.figure(figsize=(20,10))
    plt.imshow(image_display)
    plt.axis('off')
    plt.show()
