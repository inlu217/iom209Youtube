import streamlit as st

import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential


# 加载数据
df = pd.read_excel("Global.xlsx")
@st.cache_data
def load_data():
    return pd.read_excel("Global.xlsx")

def plot_average_earnings_by_category(df):
    # 计算所有类别的收入平均值
    avg_lowest_earnings = df.groupby('Category')['Lowest Monthly Earnings'].mean().reset_index()
    avg_highest_earnings = df.groupby('Category')['Highest Monthly Earnings'].mean().reset_index()

    # 绘制柱状图比较平均收入
    fig_lowest, ax_lowest = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y='Lowest Monthly Earnings', data=avg_lowest_earnings, ax=ax_lowest)
    ax_lowest.set_title('Average Lowest Monthly Earnings by Category')
    ax_lowest.set_xticklabels(avg_lowest_earnings['Category'], rotation=45)
    st.pyplot(fig_lowest)


# 设置页面配置
st.set_page_config(
    page_title="Global Youtube Statistics",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载数据
df = load_data()

# 侧边栏标题
st.sidebar.title('Global Youtube Statistics')

# 在侧边栏中选择其他条件（例如分类）
category_list = list(df.Category.unique())
selected_category = st.sidebar.selectbox('Select a category', category_list)

# 根据选择的分类筛选数据
df_filtered_by_category = df[df.Category == selected_category]

# 显示词云图
def generate_wordcloud(df):
    youtuber_counts = df['Youtuber'].value_counts().to_dict()
    wc = WordCloud(width=800, height=500, background_color='white').generate_from_frequencies(youtuber_counts)
    return wc

# 显示词云图
if st.sidebar.button('Show By Category'):
    #词云图
    st.write(f"Word Cloud for Category '{selected_category}'")
    wc_category = generate_wordcloud(df_filtered_by_category)
    st.image(wc_category.to_array(), use_column_width=True)

    #热力图
    st.write(f"Heatmap for Category '{selected_category}'")
    heatmap_data = df_filtered_by_category.corr()
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt.gcf())
    st.title(f'Income Analysis for YouTubers in Category: {selected_category}')

    # 获取选定类别的数据
    category_data = df[df['Category'] == selected_category]

    # 绘制收入分布图
    st.subheader('Income Distribution')
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    sns.histplot(category_data['Lowest Monthly Earnings'], bins=10, kde=True, ax=axes[0])
    axes[0].set_title('Lowest Monthly Earnings Distribution')
    sns.histplot(category_data['Highest Monthly Earnings'], bins=10, kde=True, ax=axes[1])
    axes[1].set_title('Highest Monthly Earnings Distribution')
    plt.tight_layout()
    st.pyplot(fig)

def plot_comparison_of_earnings(df):
    # 计算所有类别的收入平均值
    avg_lowest_earnings = df.groupby('Category')['Lowest Monthly Earnings'].mean().reset_index()
    avg_highest_earnings = df.groupby('Category')['Highest Monthly Earnings'].mean().reset_index()

    # 绘制柱状图比较平均收入
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.barplot(x='Category', y='Lowest Monthly Earnings', data=avg_lowest_earnings, ax=axes[0])
    axes[0].set_title('Average Lowest Monthly Earnings by Category')
    axes[0].set_xticklabels(avg_lowest_earnings['Category'], rotation=45)

    sns.barplot(x='Category', y='Highest Monthly Earnings', data=avg_highest_earnings, ax=axes[1])
    axes[1].set_title('Average Highest Monthly Earnings by Category')
    axes[1].set_xticklabels(avg_highest_earnings['Category'], rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

# 计算并绘制所有类别之间的 'Lowest Monthly Earnings' 和 'Highest Monthly Earnings' 的比较
if st.sidebar.button('Compare Earnings'):
    plot_comparison_of_earnings(df)

interested_features = ['Youtuber', 'Subscribers', 'Video Views', 'Uploads', 'Category',
       'Country', 'Abbreviation','Gross Tertiary Education Enrollment (%)',
       'Unemployment Rate', 'Population', 'Urban Population', 'Year','Lowest Monthly Earnings','Highest Monthly Earnings']

# 计算相关系数矩阵
correlation_matrix = df[interested_features].corr(numeric_only=True)


# 找到除了'Lowest Monthly Earnings'和'Highest Monthly Earnings'之外的最相关的特征
most_correlated_features = correlation_matrix[
    ['Lowest Monthly Earnings', 'Highest Monthly Earnings']
].abs().sort_values(by=['Lowest Monthly Earnings', 'Highest Monthly Earnings'], ascending=False).head(5)

# 在 Streamlit 中展示结果
st.write('### Top 5 Most Correlated Features with Lowest and Highest Monthly Earnings:')
st.write(most_correlated_features)

correlation_matrix = df[interested_features].corr()

# 找到除了'Lowest Monthly Earnings'和'Highest Monthly Earnings'之外的最相关的特征
correlation_matrix = correlation_matrix.drop(['Lowest Monthly Earnings', 'Highest Monthly Earnings'], errors='ignore')
most_correlated_features = correlation_matrix[
    ['Lowest Monthly Earnings', 'Highest Monthly Earnings']
].abs().sort_values(by=['Lowest Monthly Earnings', 'Highest Monthly Earnings','Lowest Monthly Earnings', 'Highest Monthly Earnings'], ascending=False).head(5)

# 在 Streamlit 中展示结果
st.write('### Top 5 Most Correlated Features with Lowest and Highest Monthly Earnings:')
st.write(most_correlated_features)
print(most_correlated_features)

# 可视化相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(most_correlated_features, annot=True, cmap='coolwarm')
st.pyplot(plt)

# 选择特征和目标变量
features = ['Video Views', 'Subscribers', 'Uploads', 'Population', 'Urban Population']
target = ['Lowest Monthly Earnings', 'Highest Monthly Earnings']

X = df[features]
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
model = Sequential()
model.add(Dense(10, input_dim=len(features), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2))  # 两个输出神经元，每个目标变量一个

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=10)

# 评估模型
loss = model.evaluate(X_test, y_test)
st.write(f'Test loss: {loss}')

# 进行预测
predictions = model.predict(X_test)
st.write('Predictions:', predictions)

