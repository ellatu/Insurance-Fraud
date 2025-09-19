import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

#此模型可用欠采样
# 假设你的数据集是一个CSV文件
data = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3_清理.csv")

# 查看数据集的前几行
print(data.head())

# 删除包含缺失值的行
data_cleaned = data.dropna()

# 如果有分类变量，进行编码
# data = pd.get_dummies(data, columns=['categorical_column'])

# 假设目标变量是 'fraud' 列
X = data.drop(['FRAUDE'], axis=1)  # 特征
y = data['FRAUDE']  # 目标变量

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化SVM分类器
svm_model = SVC(kernel='rbf', random_state=42, probability=True)  # 使用RBF核函数

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# AUC-ROC（对于二元分类问题）
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]  # 获取预测概率
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f'AUC-ROC Score: {auc_score:.2f}')