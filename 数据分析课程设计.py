#!/usr/bin/env python
# coding: utf-8

# ### 数据处理

# In[1]:


# 导入包

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_data():
    """"
    获取数据  data_all是读取的未处理的所有数据，data是用来备份处理的数据
    """
    filepath = './二手房数据济南.xlsx'
    data_all = pd.read_excel(filepath)
    data = data_all
    return data


data = get_data()
print(data.head(2))


# In[3]:


def essential_information(data):
    """
    查看数据的基本信息
    """
    #     print(data.info())
    #     print("数据集的describe\n",data.describe())
    print('备份数据，数据集的维度是：{}'.format(data.shape))


essential_information(data)


# In[4]:


def drop_col(data, del_col):
    """
    删除不需要分析的列
    """
    all_col = data.columns.values.tolist()
    for i in range(len(del_col)):
        if del_col[i] in all_col:
            data.drop(del_col[i], axis=1, inplace=True)
    return data


del_col = ["house_address", "house_id", "house_structure", "house_inner_area",
           "house_building_type", "house_building_structure", "house_elevator_sytle", "house_heating",
           "house_transaction_type", "house_last_time", "house_useage", "house_property", "house_mortgage_info",
           "house_book", "house_fx_id", "house_longitude",
           "house_latitude", "city", "house_years", "house_floor", "house_area"]

drop_col(data, del_col)
print('去除不需要的列，数据集的维度是：{}'.format(data.shape))


def drop_duplicat_row(data):
    """
    删除重复行

    """
    data.drop_duplicates(keep='last', inplace=True)
    return data


print('去除重复行前，数据集的维度是：{}'.format(data.shape))
drop_duplicat_row(data)
print('去除重复行后，数据集的维度是：{}'.format(data.shape))


# In[6]:


def drop_nan(data):
    """
    删除缺失值的行
    """
    print(data.isnull().sum())
    data.dropna(axis=0, how='any', inplace=True)


drop_nan(data)
print('去除重复行，数据集的维度是：{}'.format(data.shape))
print(data.head(3))


# In[7]:


def Extract_column_fields(data):
    """
    去掉数据中的单位 和多余描述
    """
    data['unit_price'] = pd.to_numeric(data['unit_price'].str[:-4])
    data["house_rental_area"] = data["house_rental_area"].str.extract('(\d+)').astype(int)
    data["house_decoration"] = data["house_decoration"].str.replace("装修情况", "")
    data["house_layout"] = data["house_layout"].str.replace("房屋户型", "")
    data["house_orientation"] = data["house_orientation"].str.replace("房屋朝向", "")
    data['house_lasting_year'] = pd.DatetimeIndex(data['house_listing_time']).year
    data['house_lasting_month'] = pd.DatetimeIndex(data['house_listing_time']).month
    del data["house_listing_time"]
    return data


Extract_column_fields(data)
print(data.head(2))


# ### 数据的可视化分析


def house_lasting_year(data):
    """
    上市年限与单价的关系
    """
    house_year = data[['house_lasting_year', 'unit_price']].groupby('house_lasting_year')[
        'unit_price'].mean().reset_index('house_lasting_year')
    plt.figure(figsize=(10, 5))
    plt.plot(house_year['house_lasting_year'], house_year['unit_price'])
    plt.title('上市年份与单价的关系')
    plt.ylabel('元/平米')
    plt.xlabel('年份')
    plt.show()


# house_lasting_year(data)

def house_lasting_month(data):
    """
    上市月份与单价之间的关系
    """
    house_month = data[['house_lasting_month', 'total_price']].groupby('house_lasting_month')[
        'total_price'].mean().reset_index('house_lasting_month')
    plt.figure(figsize=(10, 5))
    plt.plot(house_month['house_lasting_month'], house_month['total_price'])
    plt.title('上市月份与总价的关系')
    plt.ylabel('万元')
    plt.xlabel('年份')
    plt.show()


# house_lasting_month(data)

"""
由图可见济南市2017年的二手房价下降，20年才得以回升
由图二可以看出 1月到9月出于增长，9月开始下降

"""


def area(data):
    house_count = data.groupby('house_region')['total_price'].count().sort_values(
        ascending=False).to_frame().reset_index()
    house_mean = data.groupby('house_region')['unit_price'].mean().sort_values(ascending=False).to_frame().reset_index()

    f, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(20, 25))
    sns.barplot(x='house_region', y='unit_price', palette="Greens_d", data=house_mean, ax=ax1)
    ax1.set_title('济南各大区二手房每平米单价对比', fontsize=18)
    ax1.set_xlabel('区域', fontsize=13)
    ax1.set_ylabel('每平米单价', fontsize=13)
    sns.countplot(data['house_region'], ax=ax2)
    ax2.set_title('济南各城区房源数量', fontsize=18)
    ax2.set_xlabel('区域', fontsize=13)
    ax2.set_ylabel('二手房总量', fontsize=13)
    # print(house_count)
    plt.subplots_adjust(hspace=0.3)
    sns.barplot(x='house_region', y='total_price', palette="Blues_d", data=house_count, ax=ax3)
    ax3.set_title('济南各大区二手房房屋总价', fontsize=18)
    ax3.set_xlabel('区域', fontsize=13)
    ax3.set_ylabel('房屋总价', fontsize=13)
    plt.show()


# area (data)
"""
由上面第一幅图可以看到房子单价与地区有关，其中历下地区房价最高。这与地区的发展水平、交通便利程度以及离市中心远近程度有关
由上面第二幅图可以直接看出不同地区的二手房数量，其中历城最多
由上面第三幅图可以看出济南二手房房价基本在一千万上下，很少有高于两千万的
"""


# In[10]:


def house_rental_area(data):
    """
    面积
    """
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 5))
    # 二手房的面积分布
    sns.distplot(data['house_rental_area'], bins=20, ax=ax1, color='r')  # 直方图结合核密度图
    # 二手房面积和价位的关系
    sns.regplot(x='house_rental_area', y='total_price', data=data, ax=ax2)
    plt.show()


# house_rental_area(data)
"""
由从左到右第一幅图可以看出 基本二手房面积在60-300平方米之间，其中一百平方米左右的占比更大
由第二幅看出，二手房总结与二手房面积基本成正比，和我们的常识吻合
"""


def house_decoration_unit_price(data):
    """
    可视化装修情况情况与单价之间的关.
    """
    plt.figure(figsize=(10, 5))
    house_decoration_count = data[['unit_price', 'house_decoration']].groupby(['house_decoration']).count()
    house_decoration_count.unit_price.plot(kind='bar',
                                           title='装修情况对单价的影响',
                                           rot=0,
                                           )
    plt.xticks(rotation=90)
    plt.ylabel('元/平米')
    plt.show()


# house_decoration_unit_price(data)


"""
根据第一个柱状图可以看出 未装修的毛坯房阿济格最便宜 精装房的价格最贵 符合人们的常识
根据第二个透视表可以看出各个城市的 的装修情况 的价格的高低  明显看到历下的房价最高 市中
GDP排行	区域	2020年GDP（亿元）	增速（%）
1	历下区	1910.41	4.5
2	济南高新区	1291.54	6.5
3	市中区	1059.62	7.1
4	历城区	1017.05	5.6
5	章丘区	1002.46	7.2
6	莱芜区	641.61	5
7	槐荫区	624.27	0.2
8	天桥区	564.64	2.5

"""

pd.pivot_table(data, index=[u'house_region', u'house_decoration'], values=[u'total_price', u'unit_price'])


def orientation(data):
    """
    房屋朝向情况
    """
    plt.figure(figsize=(5, 5))
    house_orientation = data.house_orientation.value_counts().head(5)
    plt.pie(house_orientation,
            labels=['南 北 ', '南', "北", "东", "西"], autopct='%1.2f%%',
            colors=('green', 'yellowgreen', "lightskyblue", "blue", "yellow"),
            startangle=90)
    plt.title('房屋朝向占比情况')
    plt.show()


# orientation(data)

"""
分析济南市二手房的 房屋朝向情况 最好是买南北朝向的，此外具体的房屋朝向要根据房子的功能属性选择
"""


def house_layout(data1):
    """
    房屋户型
    """
    f, ax1 = plt.subplots(figsize=(30, 30))
    sns.countplot(y='house_layout', data=data1, ax=ax1)
    ax1.set_title('房屋户型', fontsize=15)
    ax1.set_xlabel('数量')
    ax1.set_ylabel('户型')
    f, ax2 = plt.subplots(figsize=(30, 30))
    sns.barplot(y='house_layout', x='unit_price', data=data1, ax=ax2)
    plt.show()


# house_layout(data) 

"""
由第一幅图看出2室1厅1厨1卫最多 2室2厅1厨1卫 是主流的户型选择
由第二幅看出 室和厅的数量增加随之价格也增加，但是室和厅之间的比例要适合
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder


def Training_set_processing_data(data):
    """
    训练模型的数据处理
    """
    data[['室', '厅']] = data['house_layout'].str.extract(r'(\d+)室(\d+)厅')
    data['室'] = data['室'].astype(float)
    data['厅'] = data['厅'].astype(float)
    del data['house_layout']
    del data['house_orientation']

    d = ['历下', '市中', '高新', '', '槐荫', '天桥', '长清', '章丘', '平阴', '商河', '历城', '济阳']
    a = ['精装', '简装', '毛坯', '其他']
    data['house_region'] = data['house_region'].apply(lambda x: d.index(x))
    data['house_decoration'] = data['house_decoration'].apply(lambda x: a.index(x))


Training_set_processing_data(data)
print(data.head(3))
print('************************************************')

# #### 测试集训练集的划分


scorelist = [[],[]]
label = data.iloc[:, 2]
size=np.arange(0.9,1,0.1)
del data['unit_price']
del data['total_price']

for i in range(0, 4):

    train_x, test_x, train_label, test_label = train_test_split(data, label, test_size=size[i],random_state=42)  #
    # 20%作为测试集， random_state在需要重复试验的时候，保证得到一组一样的随机数
    #随机森林Random Forests Model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    model = RandomForestClassifier(n_estimators=100)
    model.fit( train_x , train_label )
    print('+++++++++++++++++------------===========+++++++++++++++++++++++++++')
    scorelist[1].append(model.score(test_x , test_label ))

    model = LogisticRegression()
    model.fit(train_x, train_label)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    scorelist[0].append(model.score(test_x, test_label))
    print(train_x.head(2))

# train_label.to_csv('E:/数据分析/大作业/课程设计/train_label.csv',index = False)#不保存行索引
# test_label.to_csv('E:/数据分析/大作业/课程设计/test_label.csv',index = False)

print(scorelist)
print('----------------------------------------------------')

# #### 回归模型建立

# #### 模型评估
size = np.arange(0.9, 1, 0.1)
plt.plot(size, scorelist, color="red")
plt.legend(['逻辑回归'])
plt.xlabel('训练集占比')
plt.ylabel('准确率')
plt.title('不同的模型随着训练集占比变化曲线')
plt.show()
print(1)
