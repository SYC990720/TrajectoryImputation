import torch
import torch.nn as nn
import torch.optim as optim
from model import TLSTM

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, mode, teacher_forcing_ratio=0.2):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_dim = target.shape[-1]

        hidden, cell = self.encoder(source)
        decoder_input = target[:, 0:1, :]
        outputs = torch.zeros(batch_size, target_len, target_dim)

        for t in range(target_len-1):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t+1:t+2, :] = output
            use_teacher_forcing = torch.rand(1) < teacher_forcing_ratio
            if use_teacher_forcing and t + 1 < target_len:
                decoder_input = target[:, t+1:t+2, :]
            else:
                decoder_input = output

        return outputs
    

class TrajectoryGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, flag, device):
        super(TrajectoryGenerator, self).__init__()
        if flag==1:
            self.in_lstm = TLSTM(input_size, hidden_size)
            self.out_lstm = TLSTM(input_size, hidden_size)
        else:
            self.in_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.out_lstm = nn.LSTM(output_size, hidden_size+16, batch_first=True)
        self.embedding = nn.Embedding(4, 8)
        self.in_fc = nn.Linear(hidden_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size+16, output_size)
        self.device = device

    def forward(self, source, target, mode, teacher_forcing_ratio=0.2):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_dim = target.shape[-1]
        # representation
        output, (hidden, cell) = self.in_lstm(source)
        features = output[:, -1, :]

        # random noise
        random_vector = torch.randn(batch_size, 8)
        mean = 0
        stddev = 1
        random_vector = random_vector * stddev + mean
        random_vector = random_vector.to(self.device)
        # mode feature
        mode_vector = self.embedding(mode)
        # cat
        combined = torch.cat((features, mode_vector), dim=1)
        combined = torch.cat((combined, random_vector), dim=1)
        combined = combined.unsqueeze(0)

        decoder_input = target[:, 0:1, :]

        outputs = torch.zeros(batch_size, target_len, target_dim)
        hidden = combined
        cell = combined
        for t in range(target_len - 1):
            output, (hidden, cell) = self.out_lstm(decoder_input, (hidden, cell))
            prediction = self.out_fc(output)
            outputs[:, t + 1:t + 2, :] = prediction
            use_teacher_forcing = torch.rand(1) < teacher_forcing_ratio
            if use_teacher_forcing and t + 1 < target_len:
                decoder_input = target[:, t + 1:t + 2, :]
            else:
                decoder_input = prediction

        return outputs