import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import numpy as np
from collections import defaultdict
encoder_attn_list = defaultdict(list)
decoder_attn_list = defaultdict(list)
from pandas import DataFrame


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state（64）
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        # Why in_features is this num ?

    def forward(self, input_data):
        # print("调用Encoder！")
        #print(input_data)
        # input_data: (batch_size, T - 1, input_size)
        # input_weighted: torch.Size([128, 9, 81])
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        # input_encoded: torch.Size([128, 9, 81])
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        # hidden, cell: initial states with dimension hidden_size
        # hidden,cell: (1, x.size(0), hidden_size)

        hidden = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size)
        # print(type(hidden))
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            #repeat():在张量的某个维度上复制 torch.Size([1, 128, 64])------> torch.Size([81,128, 64])------>torch.Size([128, 81, 64])
            # x ------->  torch.Size([128, 81, 137])
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2).cuda(),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2).cuda(),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)
            """将attn_weights全部设置为相等的值"""
            # x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2).cuda(),
            #                cell.repeat(self.input_size, 1, 1).permute(1, 0, 2).cuda(),
            #                torch.zeros(input_data.permute(0, 2, 1).size()).cuda()), dim=2)
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成n列，那不确定的地方就可以写成-1
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # print("attn_weights:",attn_weights)
            a_weights = attn_weights.cpu().detach().numpy()
            for i in range(self.input_size):
                encoder_attn_list[t,i].append(np.array(a_weights[:,i]).copy())

            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            # weighted_input.unsqueeze(0)：在第1维增加一个维度
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden.cuda(), cell.cuda()))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden



        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # print("调用Decoder!")
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # print(t)
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2).cuda(),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2).cuda(),
                           input_encoded.cuda()), dim=2)
            # x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2).cuda(),
            #                cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2).cuda(),
            #                torch.zeros(input_encoded.size()).cuda()), dim=2)

            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)  # (batch_size, T - 1)
            x_weights = x.cpu().detach().numpy()
            for i in range(self.T - 1):
                decoder_attn_list[t, i].append(np.array(x_weights[:, i]).copy())


            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded.cuda())[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            # print(context.shape)
            # print(y_history[:, 1].shape)
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden.cuda(), cell.cuda()))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1)) + y_history[:, -1]
