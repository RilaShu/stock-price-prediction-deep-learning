# stock-price-prediction
- **概要**<br>
针对期货交易数据，利用深度学习方法（seq2seq）进行预测,实验数据由“中农网”提供（卓尔ZHACK比赛数据）
- **代码**<br>
arima.py -- ARIMA,BaseLine<br>
lstm_keras.py -- lstm模型KERAS版本<br>
lstm_keras.py -- lstm模型TensorFlow版本<br>
seq2seq_tf.py -- seq2seq模型(即以序列预测序列)<br>
result_output.py -- 预测与输出<br>
- **数据**<br>
data -- 期货交易数据(文件较大，提供网盘下载，详见txt内)<br>
- **结果**<br>
The Mean Absolute Error is: 81.91800705907733 (Price is about 5000).<br>
![image](https://github.com/RilaShu/stock-price-prediction-deep-learning/raw/master/images/result.png)
