import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

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

# 初始化逻辑回归模型
logreg_model = LogisticRegression(random_state=42)

# 训练模型
logreg_model.fit(X_train, y_train)
output_path = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/"
np.savetxt(output_path+"d3重要性",  logreg_model.coef_, delimiter=',')  # 保存为CSV，修改
print("系数:", logreg_model.coef_)
data2 = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3重要性.csv",sep=',')
np.savetxt(output_path+"d3重要性1",  data2, delimiter=',')
# 预测
y_pred = logreg_model.predict(X_test)

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
y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]  # 获取预测概率
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f'AUC-ROC Score: {auc_score:.2f}')

#模型调优
# 定义参数网格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正则化参数
    'penalty': ['l1', 'l2'],  # 正则化类型
    'solver': ['liblinear']  # 适用于L1和L2的求解器
}

# 初始化网格搜索
grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=3, scoring='accuracy')

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'Best Parameters: {grid_search.best_params_}')

# 使用最佳参数重新训练模型
best_logreg_model = grid_search.best_estimator_