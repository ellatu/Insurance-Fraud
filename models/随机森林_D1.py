import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# 假设你的数据集是一个CSV文件
data = pd.read_csv("/Users/ellatu/Desktop/毕业论文/文献数据/数据/d1_清理_原始.csv")

# 查看数据集的前几行
print(data.head())

# 删除包含缺失值的行
data_cleaned = data.dropna()

# 假设目标变量是 'fraud_reported' 列
X = data.drop(['fraud_reported'], axis=1)  # 特征
y = data['fraud_reported']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_model = RandomForestClassifier(max_depth = 12, min_samples_split = 2,n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)

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

# 获取特征重要性
importances = rf_model.feature_importances_

# 按重要性从高到低排序
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]  # 获取降序排列的索引
sorted_importances = importances[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

# 将特征重要性可视化
plt.figure(figsize=(12, 8))  # 可选：调整图表尺寸
plt.barh(sorted_feature_names, sorted_importances)
plt.ylabel('Feature', fontsize=12)
plt.title('D1 Feature Importance')
plt.tick_params(axis='y', which='major', labelsize=4)  # 调整y轴标签字号为8
plt.show()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 初始化网格搜索
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'Best Parameters: {grid_search.best_params_}')

# 使用最佳参数重新训练模型
best_rf_model = grid_search.best_estimator_