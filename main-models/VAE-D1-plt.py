import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import gaussian_kde
from tensorflow.keras import layers, Model
from keras.models import load_model
from typing import Dict, Any

def clean_fuction(filename, label):
    # 加载数据
    filepath = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/"
    data = pd.read_csv(filepath + filename)
    data = data.dropna()

    # 分离特征和标签
    X = data.drop(columns=[label])
    y = data[label]

    original_columns = X.columns.tolist()

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled, y, X_train, X_test, y_train, y_test, original_columns

def plt_hist(data, bins, xlim=(20,80),
             hist_color='#006699', kde_color='blue',
             title='D1 VAE Reconstruction Errors', show_threshold=True):
    # 计算KDE曲线
    # 保证数据非负
    offset = data.min() - 1e-6 if data.min() < 0 else 0
    shifted_data = data - offset + 1e-6
    # 对数变换
    log_data = np.log(shifted_data)
    # 计算对数空间中的KDE
    kde_log = gaussian_kde(log_data)
    x_log = np.linspace(log_data.min(), log_data.max(), 2000)  # 对数空间的x轴坐标
    y_kde_log = kde_log(x_log)
    # 将x轴转换回原始空间
    x_original = np.exp(x_log) + offset  # 逆变换

    # 创建画布
    plt.figure(figsize=(10, 6))
    # 绘制直方图和kde曲线
    plt.hist(data, bins=bins, alpha=0.7, color=hist_color,
             edgecolor='black', density=show_threshold)
    plt.plot(x_original, y_kde_log / x_original,  # 注意密度转换公式
             color=kde_color, linewidth=2, label='KDE')
     # 设置图表属性
    plt.xlim(*xlim)
    plt.title(title, fontsize=14)
    plt.xlabel('Reconstruction Errors', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def plt_hist2(data1, data2,bins1, bins2,xlim=(20,80),
             hist_color='#006699',
             title='D1 VAE Reconstruction Errors', show_threshold=True):
    # 计算KDE曲线1
    # 保证数据非负
    offset1 = data1.min() - 1e-6 if data1.min() < 0 else 0
    shifted_data1 = data1 - offset1 + 1e-6
    # 对数变换
    log_data1 = np.log(shifted_data1)
    # 计算对数空间中的KDE
    kde_log1 = gaussian_kde(log_data1)
    x_log1 = np.linspace(log_data1.min(), log_data1.max(), 1000)  # 对数空间的x轴坐标
    y_kde_log1 = kde_log1(x_log1)
    # 将x轴转换回原始空间
    x_original1 = np.exp(x_log1) + offset1  # 逆变换

    # 计算并绘制KDE曲线2
    # 保证数据非负
    offset2 = data2.min() - 1e-6 if data2.min() < 0 else 0
    shifted_data2 = data2 - offset2 + 1e-6
    # 对数变换
    log_data2 = np.log(shifted_data2)
    # 计算对数空间中的KDE
    kde_log2 = gaussian_kde(log_data2)
    x_log2 = np.linspace(log_data2.min(), log_data2.max(), 1000)  # 对数空间的x轴坐标
    y_kde_log2 = kde_log2(x_log2)
    # 将x轴转换回原始空间
    x_original2 = np.exp(x_log2) + offset2  # 逆变换

    # 创建画布
    plt.figure(figsize=(10, 6))
    # 绘制直方图
    plt.hist(data1, bins=bins1, alpha=0.7, color=hist_color,
             edgecolor='black', density=show_threshold,label='Non-Identified')
    plt.plot(x_original1, y_kde_log1 / x_original1,  # 注意密度转换公式
             color='blue', linewidth=2)
    plt.hist(data2, bins=bins2, alpha=0.7, color='red',
             edgecolor='black', density=show_threshold,label='Fraud')
    plt.plot(x_original2, y_kde_log2 / x_original2,  # 注意密度转换公式
             color='red', linewidth=2)
     # 设置图表属性
    plt.xlim(*xlim)
    plt.title(title, fontsize=14)
    plt.xlabel('Reconstruction Errors', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

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
            layers.Dense(input_dim, activation='sigmoid')
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

if __name__ == "__main__":
    # 清理数据
    X_scaled, y, X_train, X_test, y_train, y_test ,original_columns = clean_fuction('d1_清理_原始.csv', 'fraud_reported')
    # 加载模型
    loaded_model = load_model(
        "vae_model_d1.keras",
        custom_objects={'VAE': VAE})
    #在测试集进行预测
    X_pred, _, _ = loaded_model.predict(X_test)
    # 计算测试集的重构误差
    vae_errors = np.sum(np.square(X_test - X_pred), axis=1)
    #画出重构误差图
    print(plt_hist(vae_errors, 50))

    #制作欺诈数据的重构误差表格
    y_test1 = y_test
    y_test1 = pd.DataFrame(y_test1)
    vae_errors_fraud = vae_errors
    vae_errors_fraud = pd.DataFrame(vae_errors_fraud, columns=['are'])
    vae_errors_fraud['fraud'] = y_test1['fraud_reported']
    vae_errors_fraud = vae_errors_fraud[vae_errors_fraud['fraud'] == 1]
    vae_errors_fraud = vae_errors_fraud['are'].to_numpy()
    #画出欺诈与全部预测的对比
    print(plt_hist2(vae_errors, vae_errors_fraud,50,20))

    # 设定阈值
    threshold = np.percentile(vae_errors, 95)
    high_error_indices = np.where(vae_errors > threshold)[0]
    ae_high_error_nl = np.mean(np.square(X_test[high_error_indices] - X_pred[high_error_indices]), axis=0)
    high_nl_re = pd.DataFrame({
        'feature': original_columns,  # 列名
        'recon_error': ae_high_error_nl  # 每列的重构误差
    })
    # 按值升序排序（从低到高）
    sorted_indices = np.argsort(ae_high_error_nl)[::-1]
    sorted_labels = [original_columns[i] for i in sorted_indices]
    sorted_values = [ae_high_error_nl[i] for i in sorted_indices]
    top30_labels_desc = sorted_labels[:30][::-1]
    top30_values_desc = sorted_values[:30][::-1]

    # VAE 变量重要性
    plt.figure(figsize=(10, 6))
    plt.bar(top30_labels_desc, top30_values_desc, color='#006699')
    plt.xlabel("Feature Name")
    plt.ylabel("Reconstruction Error")
    plt.title("D1 Feature Importance(upper-tail)")
    plt.xticks(rotation=90)
    plt.show()

    # 设定阈值（例如95%分位数）
    threshold = np.percentile(vae_errors, 5)
    low_error_indices = np.where(vae_errors < threshold)[0]
    ae_low_error_nl = np.mean(np.square(X_test[low_error_indices] - X_pred[low_error_indices]), axis=0)
    # 将重构误差数组转换为带列名的Series或DataFrame
    low_nl_re = pd.DataFrame({
        'feature': original_columns,  # 列名
        'recon_error': ae_low_error_nl  # 每列的重构误差
    })
    # 按值升序排序（从低到高）
    sorted_indices = np.argsort(ae_low_error_nl)[::-1]
    sorted_labels = [original_columns[i] for i in sorted_indices]
    sorted_values = [ae_low_error_nl[i] for i in sorted_indices]
    top30_labels_desc = sorted_labels[:30][::-1]
    top30_values_desc = sorted_values[:30][::-1]

    # VAE 变量重要性
    plt.figure(figsize=(10, 6))
    plt.bar(top30_labels_desc, top30_values_desc, color='#006699')
    plt.xlabel("Feature Name")
    plt.ylabel("Reconstruction Error")
    plt.title("D1 Feature Importance(lower-tail)")
    plt.xticks(rotation=90)
    plt.show()

    #np.savetxt(output_path+"文件名", vae_errors, delimiter=',')  # 保存为CSV，修改