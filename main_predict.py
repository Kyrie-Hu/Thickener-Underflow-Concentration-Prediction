import json
import os
import sys
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from modules import Encoder, Decoder
from utils import numpy_to_tvar
import utils
from custom_types import TrainData
from constants import device
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from pandas import Series, DataFrame
from collections import defaultdict
from modules import encoder_attn_list,decoder_attn_list



def preprocess_data(dat, col_names, scale, train_num) -> TrainData:

    dat = dat[train_num:]
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]



    return TrainData(feats, targs)


def predict(encoder, decoder, t_dat, batch_size: int, T: int) -> np.ndarray:
    y_pred = np.zeros((t_dat.feats.shape[0] - T + 1, t_dat.targs.shape[1]))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            idx = range(b_idx, b_idx + T - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = encoder(numpy_to_tvar(X))
        y_pred[y_slc] = decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred

if __name__ == "__main__":
    debug = False
    save_plots = True
    # i = sys.argv[1]


    with open(os.path.join("data", "enc_kwargs.json"), "r") as fi:
        enc_kwargs = json.load(fi)
    enc = Encoder(**enc_kwargs)
    enc.load_state_dict(torch.load(os.path.join("data", "encoder"+"%s"%i+".torch"), map_location=device))

    with open(os.path.join("data", "dec_kwargs.json"), "r") as fi:
        dec_kwargs = json.load(fi)
    dec = Decoder(**dec_kwargs)
    dec.load_state_dict(torch.load(os.path.join("data", "decoder"+"%s"%i+".torch"), map_location=device))

    scaler = joblib.load(os.path.join("data", "scaler"+"%s"%i+".pkl"))
    #scaler = StandardScaler().fit()
    raw_data = pd.read_csv(os.path.join("data_foreach", "all.csv"), nrows=100 if debug else None)
    targ_cols = ("11",)


    # predict time series use test dataset
    train_num = int(len(raw_data) * 0.7)

    data = preprocess_data(raw_data, targ_cols, scaler, train_num)



    with open(os.path.join("data", "da_rnn_kwargs.json"), "r") as fi:
        da_rnn_kwargs = json.load(fi)
    final_y_pred = predict(enc.cuda(), dec.cuda(), data, **da_rnn_kwargs)



    plt.figure()
    plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
    plt.plot(final_y_pred, label='Predicted')
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted_reloaded.png", save_plots)



    #inverser transform
    X1 = scaler.inverse_transform(np.concatenate((data.feats[(da_rnn_kwargs["T"]-1):,:4],final_y_pred,data.feats[(da_rnn_kwargs["T"]-1):,4:]),axis=1))
    final_y_pred_1 = X1[:,4]
    raw_data = raw_data[train_num:]
    raw_data = raw_data.reset_index(drop=True)

    plt.figure()
    plt.plot(raw_data[['11']][(da_rnn_kwargs["T"]-1):], label="True")
    plt.plot(final_y_pred_1, label='Predicted')
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted_reloaded_1.png", save_plots)



    y_pred = np.array(final_y_pred_1)
    y_true = np.array(raw_data[['11']][(da_rnn_kwargs["T"]-1):])
    trainScore = math.sqrt(mean_squared_error(y_pred, y_true))
    print('Train Score: %f RMSE' % (trainScore))
    trainScore_mae = mean_absolute_error(y_pred, y_true)
    print('Train Score: %f MAE' % (trainScore_mae))
    trainScore_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Train Score: %f MAPE' % (trainScore_mape))
    trainScore_rmlse = np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true),2)))
    print('Train Score: %f RMLSE' % (trainScore_rmlse))

    predicted = DataFrame(final_y_pred_1)
    predicted.to_csv("darnn.csv", index=False)
