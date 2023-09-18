import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler , MinMaxScaler, StandardScaler

#%%Info
df = pd.read_csv("C:/Users/Lucas/Downloads/archive/dataset_olympics.csv",sep=",",encoding="utf-8")
df_country = pd.read_csv("C:/Users/Lucas/Downloads/archive/noc_region.csv",sep=",",encoding="utf-8")

df.sample(10)
df_country.sample(10)

df.info()
df_country.info()

#%%traitement
data=df.copy()

data.drop_duplicates(inplace=True)

df.isnull().sum()

sns.kdeplot(data=data,x="Age",shade=True)

data['Age'].describe()

sns.kdeplot(data=data,x=data['Age'].fillna(data['Age'].mean()),shade=True)

sns.kdeplot(data=data,x=data['Age'].fillna(method="ffill"),shade=True)

data['Age'].fillna(method="ffill",inplace=True)

data['Height'].describe()

sns.kdeplot(data=data,x='Height',shade="True")

sns.kdeplot(data=data,x=data['Height'].fillna(data['Height'].mean()),shade=True)

sns.kdeplot(data=data,x=data['Height'].fillna(method="ffill"),shade="True")

data['Height'].fillna(method="ffill",inplace=True)

data['Weight'].describe()

sns.kdeplot(data=data,x='Weight',shade=True)

sns.kdeplot(data=data,x=data['Weight'].fillna(method='bfill'),shade=True)

data['Weight'].fillna(method='bfill',inplace=True)

data['Medal'].fillna("No_Medal",inplace=True)

data.isnull().sum()

data.info()

#%%Outliers
data.plot(kind = "box" , subplots = True , figsize = (10,15) , layout = (3,2))

#
norm_upper_limit = data["Age"].mean() + 3 * data["Age"].std()
norm_lower_limit = data["Age"].mean() - 3 * data["Age"].std()

df_normal_new2 = data[(data["Age"] > norm_lower_limit) & (data["Age"] < norm_upper_limit)]

plt.figure(figsize=(8,5))
plt.suptitle("Distribution after Trimming",fontsize=18)
plt.subplot(1,2,1)
sns.kdeplot(data = df_normal_new2['Age'])
plt.subplot(1,2,2)
sns.boxplot(data = df_normal_new2['Age'], palette="magma")
plt.tight_layout()
plt.show()

#Height
data=df_normal_new2.copy()

norm_upper_limit = data["Height"].mean() + 3 * data["Height"].std()
norm_lower_limit = data["Height"].mean() - 3 * data["Height"].std()

df_normal_new3 = data[(data["Height"] > norm_lower_limit) & (data["Height"] < norm_upper_limit)]

plt.figure(figsize=(8,5))
plt.suptitle("Distribution after Trimming",fontsize=18)
plt.subplot(1,2,1)
sns.kdeplot(data = df_normal_new3['Height'])
plt.subplot(1,2,2)
sns.boxplot(data = df_normal_new3['Height'], palette="Greens")
plt.tight_layout()
plt.show()

#Weigth
norm_upper_limit = data["Weight"].mean() + 3 * data["Weight"].std()
norm_lower_limit = data["Weight"].mean() - 3 * data["Weight"].std()

df_normal_new1 = data[(data["Weight"] > norm_lower_limit) & (data["Weight"] < norm_upper_limit)]

plt.figure(figsize=(8,5))
plt.suptitle("Distribution after Trimming",fontsize=18)
plt.subplot(1,2,1)
sns.kdeplot(data = df_normal_new1['Weight'])
plt.subplot(1,2,2)
sns.boxplot(data = df_normal_new1['Weight'], palette="Reds")
plt.tight_layout()
plt.show()

#%%Data Visualization
data

#Q1 : What is the rate between Male to Female that partispate at the olympic ?

da1=data['Sex'].value_counts().reset_index(name='count')
da1

plt.figure(figsize = (3,3))
size=0.6
plt.title("Rate between Male to Female")
da1["count"].plot.pie(autopct='%0.0f%%',labels=['Male','Female'],colors=['#1E90FF','#FFB6C1'],wedgeprops=dict(width=size, edgecolor='w'))

#Q2 : what is rate of female to male that win with Medal

da2=data.groupby('Sex')['Medal'].value_counts().reset_index(name='count')
da2

fig = px.sunburst(data, path=['Sex','Medal']).update_traces(textinfo='label+percent parent')
fig.show()

#ALots of participants male and female not win medal at olympic

#Q3 : What is number of participants per year ?

data['Year'].unique()

plt.figure(figsize=(6,6))
sns.countplot(data=data,y="Year")

#1992 is heigher

#Q4 : What is the most year participants win Gold Medal ?

da3=data[data['Medal']=='Gold'] ['Year'].value_counts().reset_index(name='count')
da3.set_axis(['Year','Count'],axis='columns',inplace=True)

sns.lineplot(data=da3,x="Year",y='Count')

#2016 has heigher Gold Medal

#Q5 : What is heigher sport that sporters paly it?

plt.figure(figsize=(6,12))
sns.countplot(data=data,y='Sport')

#-Athletics










