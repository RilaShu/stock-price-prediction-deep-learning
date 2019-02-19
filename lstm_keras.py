# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error


def read_csv_file(filename):
    """
    read csv file
    :param filename:
    :return:
    """
    data = pd.read_csv(filename, sep=',', header=0)
    return data


# 读数据
df = read_csv_file('data/all.csv')
# 取其中的有用项
input_data = df.loc[:, ['StartPrice', 'EndPrice', 'HighPrice', 'LowPrice', 'Total_ContNum', 'AvePrice']].values
output_data = df.loc[:, ['AvePrice']].values

# 绘制均价走势
plt.plot(output_data, label="Ave")
plt.legend()
plt.show()

# 归一化
min_max_scaler = MinMaxScaler()
input_data = min_max_scaler.fit_transform(input_data)
scaler_for_output = MinMaxScaler(feature_range=(0, 1))
output_data = scaler_for_output.fit_transform(output_data)#scaler_for_output将被用于还原
# 窗口长度
window_len = 10
# 划分train和test
normalized_train_data = input_data[0:-(2*window_len+1)]
normalized_test_data = input_data[-(2*window_len+1):]
label_train = output_data[0:-(2*window_len+1)]
label_test = output_data[-(2*window_len+1):]

# 训练集
train_x = []
for i in range(len(normalized_train_data)-window_len):
    temp_set = normalized_train_data[i:(i + window_len)].copy()
    train_x.append(temp_set)
train_x = np.array(train_x)
train_y = np.array(label_train[window_len:])
# 测试集
test_x = []
for i in range(len(label_test) - window_len):
    temp_set = normalized_test_data[i:(i + window_len)].copy()
    test_x.append(temp_set)
test_x = np.array(test_x)
test_y = np.array(label_test[window_len:])


# model
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.10, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, activation='tanh', input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# initialise model architecture
nn_model = build_model(train_x, output_size=1, neurons=32)
# model output is next price normalised to 10th previous closing price
# train model on data
# note: eth_history contains information on the training error per epoch
nn_history = nn_model.fit(train_x, train_y,
                            epochs=3, batch_size=1, verbose=2, shuffle=True)
print(nn_model.predict(test_x))

plt.plot(scaler_for_output.inverse_transform(test_y), label="actual")
plt.plot(scaler_for_output.inverse_transform(nn_model.predict(test_x)), label="predicted")
plt.legend()
plt.show()
MAE = mean_absolute_error(test_y, nn_model.predict(test_x))
print('The Mean Absolute Error is: {}'.format(MAE))


# 连续预测
def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


predictions = predict_sequence_full(nn_model, test_x, 10)
plt.plot(scaler_for_output.inverse_transform(test_y), label="actual")
plt.plot(scaler_for_output.inverse_transform(predictions), label="predicted")
plt.legend()
plt.show()
MAE = mean_absolute_error(test_y, predictions)
print('The Mean Absolute Error is: {}'.format(MAE))

