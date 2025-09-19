import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#此模型可用欠采样
# 假设你的数据集是一个CSV文件
data_1 = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3数据集/anomaly_raw_file_Dic17.csv",sep=';')
data_2 = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3数据集/normal_raw_file_Dic17.csv",sep=';')
data = pd.concat([data_1, data_2], axis=0, ignore_index=True)

# 查看数据集的前几行
print(data.head())

# 删除包含缺失值的行
data_cleaned = data.dropna()

# 如果有分类变量，进行编码
# data = pd.get_dummies(data, columns=['categorical_column'])

# 假设目标变量是 'fraud' 列
X = data_cleaned.drop(['FRAUDE'], axis=1)  # 特征

scaler = StandardScaler()
X_scaled1 = scaler.fit_transform(X)

# 初始化孤立森林模型
iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)  # contamination参数表示异常点的比例

# 训练模型
iso_forest.fit(X)

# 预测异常点（-1表示异常，1表示正常）
anomaly_labels = iso_forest.predict(X)

# 将预测结果转换为0（正常）和1（异常）
anomaly_labels = [1 if label == -1 else 0 for label in anomaly_labels]

# 将异常标签添加到原始数据集中
data_cleaned['anomaly'] = anomaly_labels

# 查看异常检测结果
print(data_cleaned['anomaly'].value_counts())

# 假设目标变量是 'fraud'
y_true = data_cleaned['FRAUDE']
y_pred = anomaly_labels

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 分类报告
class_report = classification_report(y_true, y_pred)
print('Classification Report:')
print(class_report)

# 获取异常分数（分数越低，越可能是异常）
anomaly_scores = iso_forest.decision_function(X)

# 计算AUC-ROC
auc_score = roc_auc_score(y_true, -anomaly_scores)  # 取负号，因为分数越低表示异常
print(f'AUC-ROC Score: {auc_score:.2f}')

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, -anomaly_scores)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Isolation Forest')
plt.legend()
plt.show()

# 按异常标签分组，查看异常点的统计信息
anomaly_summary = data_cleaned.groupby('anomaly').mean()
print(anomaly_summary)

# 查看异常点中欺诈样本的比例
fraud_in_anomalies = data_cleaned[data_cleaned['anomaly'] == 1]['FRAUDE'].mean()
print(f'Fraud Rate in Anomalies: {fraud_in_anomalies:.2f}')

# 尝试不同的contamination值
iso_forest_tuned = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)  # 假设异常点比例为10%
iso_forest_tuned.fit(X)

# 预测异常点
anomaly_labels_tuned = iso_forest_tuned.predict(X)
anomaly_labels_tuned = [1 if label == -1 else 0 for label in anomaly_labels_tuned]

# 评估调整后的模型
conf_matrix_tuned = confusion_matrix(y_true, anomaly_labels_tuned)
print('Tuned Confusion Matrix:')
print(conf_matrix_tuned)