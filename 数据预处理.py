import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE



# 导入数据
filepath= "/Users/ellatu/Desktop/毕业论文/文献数据/数据/"
d1 = pd.read_csv(filepath+"insurance_claims.csv")
d2 = pd.read_csv(filepath+"creditcard.csv")
data_1 = pd.read_csv(filepath+"d3数据集/anomaly_raw_file_Dic17.csv",sep=';')
data_2 = pd.read_csv(filepath+"d3数据集/normal_raw_file_Dic17.csv",sep=';')
d3 = pd.concat([data_1, data_2], axis=0, ignore_index=True)
print(d3.nunique())
y1 = d1['fraud_reported']
y2 = d2['Class']
X2 = d2.drop(['Class'], axis=1)
# 删除包含缺失值的行
d3 = d3.dropna()
y3 = d3['FRAUDE']
X3 = d3.drop([ 'FRAUDE','id_siniestro'], axis=1)
print(y1)

#查看数据空值数量
total1 = d1.isnull().sum().sort_values(ascending = False)
percent = (d1.isnull().sum()/d1.isnull().count()*100).sort_values(ascending = False)
print(pd.concat([total1, percent], axis=1, keys=['Total', 'Percent']).transpose())

total2 = d2.isnull().sum().sort_values(ascending = False)
percent = (d2.isnull().sum()/d2.isnull().count()*100).sort_values(ascending = False)
print(pd.concat([total2, percent], axis=1, keys=['Total', 'Percent']).transpose())

total3 = d3.isnull().sum().sort_values(ascending = False)
percent = (d3.isnull().sum()/d3.isnull().count()*100).sort_values(ascending = False)
print(pd.concat([total3, percent], axis=1, keys=['Total', 'Percent']).transpose())

#替换d1的问号
d1.replace('?', np.nan, inplace = True)
#缺失值
print(d1.isna().sum())
ax = msno.bar(d1,color="#006699")
ax.tick_params(axis='x', labelsize=8)
plt.show()

#缺失值填充众数
d1['collision_type'] = d1['collision_type'].fillna(d1['collision_type'].mode()[0])
d1['property_damage'] = d1['property_damage'].fillna(d1['property_damage'].mode()[0])
d1['police_report_available'] = d1['police_report_available'].fillna(d1['police_report_available'].mode()[0])

#统计唯一值
print(d1.nunique())
#drop掉不需要的
to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']
d1.drop(to_drop, inplace = True, axis = 1)
print(d1.nunique())
#分割
X1 = d1.drop('fraud_reported', axis = 1)
#分割数字和具体的
cat_d1 = X1.select_dtypes(include = ['object'])
for col in cat_d1.columns:
    print(f"{col}: \n{cat_d1[col].unique()}\n")
cat_d1 = pd.get_dummies(cat_d1, drop_first=True)
#数字
num_d1 = X1.select_dtypes(include = ['int64'])
#合并
X1 = pd.concat([num_d1, cat_d1], axis = 1)
print(X1.head)
plt.figure(figsize=(20, 12))
plotnumber = 1

for col in X1.columns:
    if plotnumber <= 26:
        ax = plt.subplot(3, 9, plotnumber)
        ax2 = sns.distplot(X1[col])
        plt.xlabel(col, fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax2.tick_params(axis='both', labelsize=6)

    plotnumber += 1

plt.tight_layout()
plt.show()


num_d1 = X1[['age','total_claim_amount','months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
# 特征标准化
scaler = StandardScaler()
X_scaled1 = scaler.fit_transform(num_d1)
scaled_num_df = pd.DataFrame(data = X_scaled1, columns = num_d1.columns, index = X1.index)
X1.drop(columns = scaled_num_df.columns, inplace = True)
X1 = pd.concat([scaled_num_df, X1], axis = 1)
print(X1)
x1 = X1
output_path0 = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/d1_清理_原始.csv"
x1['fraud_reported'] = d1['fraud_reported']  # 添加目标变量列
#x1.to_csv(output_path0, index=False)
plt.figure(figsize = (14,14))
plt.title('D1 features correlation plot (Pearson)')
corr1 = x1.corr()
ax = sns.heatmap(corr1,xticklabels=corr1.columns,yticklabels=corr1.columns,linewidths=.1,cmap="Reds")
ax.tick_params(axis='both', labelsize=8)
plt.show()

#--------------------------------------------------------------------------
print(y1)
# 初始化过采样器
smote = SMOTE(sampling_strategy={1: 750}, random_state=42)
# 应用过采样
X1_resampled, y1_resampled = smote.fit_resample(X1, y1)

# 查看平衡后的类别分布
print("原始类别分布:", {0: sum(y1 == 0), 1: sum(y1 == 1)})
print("过采样后分布:", {0: sum(y1_resampled == 0), 1: sum(y1_resampled == 1)})

# 将过采样后的数据转换为 DataFrame
resampled_data1 = pd.DataFrame(X1_resampled, columns=X1.columns)  # 特征
resampled_data1['fraud_reported'] = y1_resampled  # 添加目标变量列


#------------------------------------------------------------------------
plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = d2.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()

# 查看类别分布
print("原始数据类别分布:", Counter(y2))

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)

# 欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_scaled, y2)


# 将欠采样后的数据转换为 DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X2.columns)  # 特征
resampled_data['Class'] = y_resampled  # 添加目标变量列


#---------------------------------------------------------------------
# 查看类别分布
print("原始数据类别分布:", Counter(y3))

# 特征标准化
scaler = StandardScaler()
X3_scaled = scaler.fit_transform(X3)

# 欠采样
rus = RandomUnderSampler(random_state=42)
X3_resampled, y3_resampled = rus.fit_resample(X3_scaled, y3)

# 将欠采样后的数据转换为 DataFrame
resampled_data3 = pd.DataFrame(X3_resampled, columns=X3.columns) # 特征
print(resampled_data3)
resampled_data3['FRAUDE'] = y3_resampled  # 添加目标变量列

# 输出到新的 CSV 文件
output_path1 = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/d1_清理.csv"
output_path2 = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/d2_清理.csv"
output_path3 = "/Users/ellatu/Desktop/毕业论文/文献数据/数据/d3_清理.csv"
X1['fraud_reported'] = d1['fraud_reported']  # 添加目标变量列
resampled_data1.to_csv(output_path1, index=False)
resampled_data.to_csv(output_path2, index=False)
resampled_data3.to_csv(output_path3, index=False)

