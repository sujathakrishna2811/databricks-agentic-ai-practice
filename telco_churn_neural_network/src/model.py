import torch.nn as nn


class TelcoChurnNN(nn.Module):
    """
    Feed-forward neural network for binary Telco churn classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
    ):
        super().__init__()

        self.layer1 = nn.Linear(
            input_size,
            hidden_size_1,
        )

        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(
            hidden_size_1,
            hidden_size_2,
        )

        self.relu2 = nn.ReLU()

        self.output_layer = nn.Linear(
            hidden_size_2,
            1,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        return self.output_layer(x)