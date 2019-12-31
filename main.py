import typing
from typing import Tuple
import json
import os
import sys
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")


def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    # print(mask.shape)
    # print(mask)
    dat_cols = list(dat.columns)
    #print(dat_cols)
    #print(col_names)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    #print(mask)

    # feats:取非target列
    # targs:取target列
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    #print(feats.shape)
    #print(feats.shape)

    return TrainData(feats, targs), scale

"""
神经网络的典型处理如下：
1、定义可学习参数的网络结构（堆叠各层和层的设计）
2、数据集的输入
3、对输入进行处理（由定义的网络层进行处理），主要体现在网络的前向传播
4、计算loss，由loss层进行计算
5、反向传播求梯度
6、根据梯度改变参数值，最简单的实现方式是
（SGD）weight = weight - learning_rate * gradient
"""

def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

    #print(learning_rate)
    #print("train_data:",train_data)
    #print("TrainData:",TrainData)
    #print(n_targs)
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    #print("train_cfg:",train_cfg)
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}

    """"
    Encoder传入参数：
    input size: number of underlying factors (81)
    T: number of time steps (10)
    hidden_size: dimension of the hidden state
    """
    encoder = Encoder(**enc_kwargs).to(device)
    #print("encoder:",encoder)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}

    """"
        Decoder传入参数：
        encoder_hidden_size: dimension of the encoder(64)
        decoder_hidden_size:dimension of the decoder(64)
        T: number of time steps (10)
        out_feats: number of the output feature(1)
    """

    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    # np.ceil： 计算大于等于该值的最小整数
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        # np.random.permutation: 随机排列一个序列，返回一个排列的序列。
        # np.random.permutation(10)
        # array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        # print("perm_idx:",perm_idx) #23892的一个排列

        # 从0到t_cfg.train_size，步长为t_cfg.batch_size
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)] # 在perm_idx中以batch_size为单位选取数据，即为batch_idx
            #print("batch_idx:",batch_idx)

            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 2 == 0:
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


"""
构造出batch_size个9行81列的数组feats，作为预训练的特征数据
构造出batch_size个9行1列的数组y_history,作为预训练的目标历史数据
构造出batch_size个1列的数组y_target，作为预训练的目标预测数据
指的注意的是，feats，y_history均为有序数组，即没有破坏训练数据的时序信息
"""
def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    # feats.shape: (128, 9, 81) 128个9行81列的全零数组
    #print("feats.shape:",feats.shape)
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    # y_history: (128, 9, 81) 128个9行1列的数组
    y_target = train_data.targs[batch_idx + t_cfg.T]
    #y_target.shape: (128, 1) 需要预测的10步长之后的目标变量，128维
    # print("y_target:",y_target)
    # print("y_target.shape:", y_target.shape)

    # b_i, b_idx ----> (0, seq[0]), (1, seq[1]), (2, seq[2])
    for b_i, b_idx in enumerate(batch_idx):  # 遍历这个无序的数组
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)  # 截取这个区间的值
        feats[b_i, :, :] = train_data.feats[b_slc, :] # 构造出batch_size个9行81列数组
        y_history[b_i, :] = train_data.targs[b_slc] # 构造出batch_size个9行1列数组

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    #在此处调用encoder()函数
    #numpy_to_tvar： 将numpy类型转换为tensor类型
    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    # 在此处调用decoder()函数
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        # print("range(len(y_pred))[y_slc]:",range(len(y_pred))[y_slc])
        batch_idx = range(len(y_pred))[y_slc]
        #print("batch_idx:",batch_idx)
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        #print("X.shape:",X.shape())
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


if __name__ == "__main__":

    save_plots = True
    debug = False

    # i = sys.argv[1]
    # print("i",i)
    i = 5
    print("i", i)

    raw_data = pd.read_csv(os.path.join("data_foreach", "res_all_selected_features_smooth_"+"%s"%i+".csv"), nrows=100 if debug else None)
    logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
    targ_cols = ("11",)
    data, scaler = preprocess_data(raw_data, targ_cols)

    # print(data)
    # print(scaler)
    da_rnn_kwargs = {"batch_size": 128, "T": 10}

    """
    da_rnn 传入参数 
    data:包括feats和targets
    n_targs:预测目标变量的数量
    learning_rate:学习率，传入时为0.001，但是没有接受传入值，使用了自定义值0.01
    batch_size :128
    时间步长T：10
    """

    # **da_rnn_kwargs 传入字典型数据
    config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=.001, **da_rnn_kwargs)
    # print(config)
    # print(model)

    """"train()函数调用了model，即调用了da_rnn函数，da_rnn函数调用了Encoder类,实例化Encoder类时(train_iteration)，就会调用其forward函数"""
    iter_loss, epoch_loss = train(model, data, config, n_epochs=4, save_plots=save_plots)
    # final_y_pred
    final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

    plt.figure()
    # plt.semilogy： 在y轴上绘制具有对数缩放的图
    plt.semilogy(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    plt.figure()
    plt.plot(final_y_pred, label='Predicted')
    plt.plot(data.targs[config.train_size:], label="True")
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted.png", save_plots)

    with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
        json.dump(da_rnn_kwargs, fi, indent=4)

    #scikit-learn的模型持久化的操作，导入joblib，通过joblib的dump可以将模型保存到本地
    #通过joblib的load方法，加载保存的模型。clf = joblib.load("train_model.m")   clf.predit(test_X)

    joblib.dump(scaler, os.path.join("data", "scaler"+"%s"%i+".pkl"))
    #torch保存模型参数
    torch.save(model.encoder.state_dict(), os.path.join("data", "encoder"+"%s"%i+".torch"))
    torch.save(model.decoder.state_dict(), os.path.join("data", "decoder"+"%s"%i+".torch"))
