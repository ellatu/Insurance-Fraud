import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

#清理程序
def clean_fuction(filename,label1,label2):
    # 加载数据
    filepath = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/"
    data = pd.read_csv(filepath + filename)

    # 分离特征和标签
    X = data.drop(columns=[label1,label2])
    y = data[label1]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled,y,X_train, X_test, y_train, y_test

#AE模型实现
class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim=32,**kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.input_dim = input_dim  # 必须显式保存为实例变量
        self.latent_dim = latent_dim  # 必须显式保存为实例变量
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='tanh'),
            layers.Dense(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='tanh'),
            layers.Dense(input_dim, activation='relu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded#解码所得

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim
        })
        return config

def training(X_train,model,epochs,title):
    # 训练并评估AE
    Model = model(input_dim = X_scaled.shape[1])
    Model.compile(optimizer='adam', loss='mse')
    history = Model.fit(
        X_train,X_train,
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        verbose=1  # 显示进度条和指标
    )
    # 训练完成后可直接绘制历史记录
    plt.plot(history.history['loss'], label='train_loss')
    #plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()

    Model.save("ae_model_d2_4.keras")
    return


# 主程序
if __name__ == "__main__":
    # 清理数据
    X_scaled, y, X_train, X_test, y_train, y_test = clean_fuction('creditcard.csv', 'Class','Time')
    training(X_train,Autoencoder, 100, 'AE')

