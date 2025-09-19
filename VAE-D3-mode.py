import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers, Model
from typing import Dict, Any

#清理程序
def clean_fuction(filename1,filename2,label1,label2):
    # 加载数据
    filepath = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3数据集/"
    data1 = pd.read_csv(filepath + filename1, sep=';')
    data2 = pd.read_csv(filepath + filename2, sep=';')
    data = pd.concat([data1, data2], axis=0, ignore_index=True)
    data = data.dropna()

    # 分离特征和标签
    X = data.drop(columns=[label1,label2])
    y = data[label1]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled,y,X_train, X_test, y_train, y_test

#VAE模型实现
class VAE(Model):
    def __init__(self, input_dim, latent_dim=32,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 编码器5层
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation='tanh', dtype=tf.float32),
            layers.Dropout(0.2),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='tanh'),
            layers.Dense(2 * latent_dim)  # 输出均值和方差
        ])

        # 解码器5层
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

    def sample(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def call(self, x):
        # 编码
        z_params = self.encoder(x)
        z_mean, z_log_var = tf.split(z_params, 2, axis=1)
        z = self.sample(z_mean, z_log_var)

        # 解码
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def get_config(self) -> Dict[str, Any]:
        # 获取父类的配置并添加自定义参数
        config = super().get_config().copy()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        # 明确处理父类的参数
        parent_config = {k: v for k, v in config.items() if k in ["name", "trainable", "dtype"]}
        child_config = {k: v for k, v in config.items() if k not in ["name", "trainable", "dtype"]}
        return cls(**child_config,  ** parent_config)

def training_vae(data,epoch):
    model = VAE(input_dim = data.shape[1])
    optimizer = optimizers.Adam(learning_rate=1e-3)
    train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(128)
    # 记录训练过程
    train_loss_history = []

    for epoch in range(epoch):
        epoch_loss = 0.0  # 每个epoch的累计损失
        total_loss = 0
        for batch in train_dataset:
            # 检查批次形状
            with tf.GradientTape() as tape:
                # 前向传播
                reconstructed, z_mean, z_log_var = model(batch)
                # 计算重构损失
                reconstruction_loss = tf.reduce_mean( tf.reduce_sum(tf.square(batch - reconstructed), axis=1))
                # 计算KL散度
                kl_per_sample = -0.5 * tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp( z_log_var),
                    axis=1
                )
                kl_loss = tf.reduce_mean(kl_per_sample)
                # 总损失
                total_loss = reconstruction_loss + kl_loss

            # 计算梯度
            gradients = tape.gradient(total_loss, model.trainable_variables)
            # 应用梯度更新参数
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 累加损失
            epoch_loss += total_loss.numpy()
        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataset)
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return model


if __name__ == "__main__":
    # 清理数据
    X_scaled, y, X_train, X_test, y_train, y_test = clean_fuction('anomaly_raw_file_Dic17.csv', 'normal_raw_file_Dic17.csv','FRAUDE','id_siniestro')
    X_train = X_train.astype(np.float32)
    #训练函数
    vae = training_vae(X_train, 50)
    vae.save("vae_model_d3_relu.keras")
