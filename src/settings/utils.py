from matplotlib import pyplot as plt

from torch import Tensor


def show_tensor_image(tensor_image: Tensor) -> None:
    """
    Displays tensor as image
    """
    plt.imshow(tensor_image[0])
    plt.axis('off')
