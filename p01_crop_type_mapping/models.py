"""Deep learning models for field time series. THIS FILE IS YOUR PRACTICE AREA.

Every model takes x shaped (batch, months, channels) and returns logits shaped
(batch, n_classes). The training loop in step4_train_deep.py handles the rest.

Check your work with:   python -m pytest p01_crop_type_mapping/tests -q
Then train it with:     python -m p01_crop_type_mapping.step4_train_deep --model tempcnn
"""
import torch
from torch import nn


class LinearBaseline(nn.Module):
    """Reference model (already done): flatten the series, one linear layer.

    Use it to check the training loop works. Your models should beat it.
    """

    def __init__(self, n_channels: int, n_months: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_channels * n_months, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class TempCNN(nn.Module):
    """EXERCISE 2: Temporal CNN (Pelletier et al., 2019).

    Idea: 1D convolutions slide over *time*, so the model learns short seasonal
    patterns (green-up, harvest drop) wherever they occur in the year.

    Suggested architecture:
      3 x [Conv1d(kernel_size=5, padding=2) -> BatchNorm1d -> ReLU -> Dropout(0.2)]
      with `hidden` filters, then Flatten -> Linear(hidden * n_months, 128)
      -> ReLU -> Dropout -> Linear(128, n_classes)

    Hint: Conv1d expects (batch, channels, time), but x is (batch, time, channels).
          Look up tensor.transpose(1, 2).
    """

    def __init__(self, n_channels: int, n_months: int, n_classes: int, hidden: int = 64):
        super().__init__()
        # TODO: build the layers here
        raise NotImplementedError("Exercise 2: implement TempCNN.__init__")

    def forward(self, x):
        # TODO: transpose, run conv blocks, flatten, classify
        raise NotImplementedError("Exercise 2: implement TempCNN.forward")


class LSTMClassifier(nn.Module):
    """EXERCISE 3: recurrent model.

    Suggested architecture:
      nn.LSTM(n_channels, hidden, num_layers=2, batch_first=True, dropout=0.2)
      then classify from the last time step's output with a Linear layer.

    Stretch goals once it works:
      - bidirectional=True (remember the output size doubles)
      - average over all time steps instead of taking only the last one
    """

    def __init__(self, n_channels: int, n_months: int, n_classes: int, hidden: int = 64):
        super().__init__()
        raise NotImplementedError("Exercise 3: implement LSTMClassifier.__init__")

    def forward(self, x):
        raise NotImplementedError("Exercise 3: implement LSTMClassifier.forward")


# EXERCISE 4 (stretch): a small Transformer encoder. Add a learned positional
# embedding per month, nn.TransformerEncoder with 2 layers, mean-pool over
# time, then a Linear head. Register it below as "transformer".

MODELS = {
    "linear": LinearBaseline,
    "tempcnn": TempCNN,
    "lstm": LSTMClassifier,
}


def build(name: str, n_channels: int, n_months: int, n_classes: int) -> nn.Module:
    return MODELS[name](n_channels=n_channels, n_months=n_months, n_classes=n_classes)


if __name__ == "__main__":
    # Quick smoke test: python -m p01_crop_type_mapping.models
    x = torch.randn(8, 12, 11)
    for name in MODELS:
        try:
            print(name, tuple(build(name, 11, 12, 4)(x).shape))
        except NotImplementedError as e:
            print(name, "->", e)
