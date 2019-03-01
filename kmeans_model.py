#coding:utf-8
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #导入K均值聚类算法
from sklearn.decomposition import PCA #进行降维处理

#对txt文档进行处理成标准的csv文件类型
def deal_file_type(source_data_file):
    new_file =source_data_file+'formal_data.csv'
    if not os.path.exists(new_file) :
        print('重新创建文件')
        with open(source_data_file,'r',encoding='utf-8') as fin:
            new_data_str = fin.read().replace('\t',',')
            with open(source_data_file+'formal_data.csv','w') as fout:
                fout.write(new_data_str)
                print('创建结束')
    else:
        print('文件已经存在')
    return new_file

#读取数据填充并清洗
def read_formal_data(new_file):
    filename =open(new_file)
    df = pd.read_csv(filename)
    df.columns = ['AGE', 'SEX', 'SCOMB_MSG', 'SCOMB_MSG_BXL', 'REGISTER_TIME', 'IS_SCHOOL_USER', 'FLOW_12', 'FLOW_11', 'FLOW_10', 'FLOW_09', 'FLOW_08']
    df = df.fillna(0)
    return df

#数据格式化
#利用drop_duplicates去重查找出全部字段数值型,替换之后，重新利用去重进行验证,先统一对未知或者空值置为0
def init_stand(df,columns):
    print('格式化开始')
    for column in columns:
        l_list = df[column].drop_duplicates()
        j = 1
        DISC = {} #用于替换的数据字典 如{'男':1,'女':2,'未知':0}
        for i in l_list:
            if i == '未知' or i == 0 :
                DISC[i] = 0
            else:
                DISC[i] = j
                j+=1
        df[column] = df[column].replace(DISC)
    print('格式化结束')

#数据归一化(线性归一化)
def nomalization(df,columns):
    for listName in columns:
        if df[listName].max() != df[listName].min():
            df[listName] = (df[listName]-df[listName].min())/(df[listName].max()-df[listName].min())

#对于聚类需要计算距离，且字段是离散化的，需要进行one-hot编码
def onehot(df,columns):
    print('开始进行onehot编码')
    for l in columns:
        df_dummies2 = pd.get_dummies(df[l], prefix=l) #选定要进行onehot编码的列，增名为scomb_____1
        df = df.drop(l, axis=1) #删除原有的列
        df = pd.concat([df,df_dummies2],axis=1) #重新拼接
    print('onehot编码结束')

#进行降维处理
def draw(Data,centers,index=None):
    """
            降维展示分类效果，
            不代表实际数据的分布
    """
    fig,axes = plt.subplots(1,1)#fig为幕布对像 axes为子图对象
    pca = PCA(n_components=2)#设置维度
    new_data = pca.fit_transform(Data) #进行绘图
    # print (new_data)
    axes.scatter(new_data[:,0], new_data[:,1],c='b',marker='o',alpha=0.5)#在子图0上绘制二维分布图像
    # print('-------------------------')
    # print(centers)
    # for center_index,center in enumerate(centers):#绘制分类中心点
    axes.scatter(centers[:,0], centers[:,1],c='r',marker='*',alpha=0.5)
    plt.show()


#进行kmeans聚类
def kmeans(df,k,drawplt = True):
    data_set = df.as_matrix()
    kmeans = KMeans(n_clusters=k).fit(data_set)
    centers = kmeans.cluster_centers_#获取中心点
    pca = PCA(n_components=2)  # 设置维度
    centers_d = pca.fit_transform(centers)
    labels = kmeans.labels_ #获取聚类标签
    inertia = kmeans.inertia_ #获取聚类准则的总和
    if drawplt:
        draw(data_set,centers_d)
    return inertia

def kmeans_k(list):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x=list[:,0],y=list[:,1])
    plt.show()

if __name__ == "__main__":
    source_data_file = 'test_data.txt'
    new_file = deal_file_type(source_data_file)
    df = read_formal_data(new_file)
    init_stand(df,['AGE','SEX','SCOMB_MSG','SCOMB_MSG_BXL','REGISTER_TIME','IS_SCHOOL_USER'])
    nomalization(df,['FLOW_12', 'FLOW_11', 'FLOW_10', 'FLOW_09', 'FLOW_08'])
    onehot(df,['AGE', 'SEX', 'SCOMB_MSG', 'SCOMB_MSG_BXL', 'REGISTER_TIME', 'IS_SCHOOL_USER'])
    kmeans(df,30)

'''
通过枚举，以及枚举后的图像获取最佳拐点（k值）
'''
    # list_k = []
    # for k in range(4,100):
    #     print('当前聚类数目为：',k)
    #     list_k.append([k,kmeans(df,k,drawplt=False)])
    # kmeans_k(list_k)
