from torch import nn, Tensor

from settings import model_settings

class BaseModel(nn.Module):
    """
    Basic LeNet architecture used in all experiment variants
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20000, 500),
            nn.Dropout(p=dropout_rate),
            nn.Linear(500, model_settings.NUM_CLASSES),
        )

    def forward(self, img: Tensor):
        return self._model(img)
