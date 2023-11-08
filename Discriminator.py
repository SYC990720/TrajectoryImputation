import torch
import torch.nn as nn
import torch.optim as optim


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TrajectoryDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        probabilities = torch.sigmoid(output)
        return probabilities


class TrajectoryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TrajectoryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        probabilities = torch.softmax(output, dim=1)
        return probabilities
