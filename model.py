import torch.nn as nn
import lib
import torch.nn.functional as F
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(len(lib.ws), 100)
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=lib.HIDDEN_SIZE, num_layers=lib.NUM_LAYER,
                             batch_first=True, bidirectional=lib.BIDIRECTIONAL, dropout=lib.DROPOUT)
        # self.lstm2 = nn.LSTM(input_size=2*lib.HIDDEN_SIZE, hidden_size=10, num_layers=1,
        #                      batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(2*lib.HIDDEN_SIZE, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        """
        :param x: [batch_size, max_len]
        :return:
        """
        x = self.embedding(x)  # [batch_size, max_len, 100]
        # x [batch_size, max_len, 2*hidden_size]
        # h_n [2*2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm1(x)
        output_fw = h_n[-2, :, :]  # 正向最后一次的输出
        output_bw = h_n[-1, :, :]  # 反向最后一次的输出
        # output [batch_size, hidden_size*2]
        # x, (h_n, c_n) = self.lstm2(x)   # [batch_size, max_len, 10]
        # output = x[:, -1, :]
        # output = self.fc1(output)
        # out = F.relu(output)
        # output = self.fc2(output)
        output = torch.cat([output_fw, output_bw], dim=-1)
        output = self.fc1(output)
        output = F.relu(output)
        out = self.fc2(output)

        return F.log_softmax(out, dim=-1)
