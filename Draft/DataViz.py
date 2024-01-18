import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler , MinMaxScaler, StandardScaler

#%%Info

#Import des données
df = pd.read_csv("C:/Users/Lucas/Downloads/archive/dataset_olympics.csv",sep=",",encoding="utf-8")
df_country = pd.read_csv("C:/Users/Lucas/Downloads/archive/noc_region.csv",sep=",",encoding="utf-8")

#On retire les JO d'hiver
df = df[(df['Year'] >= 1994) & (df['Year'] % 4 == 0)]

#Vu 'rapide' des données
df.sample(10)
df_country.sample(10)

#On regarde les différents sports possibles
df['Sport'].unique()
df['Games'].unique()

#On regarde le nb d'observations pour les 10 premiers sport
df['Sport'].value_counts()[0:10]

#Visualisation du nombre de médailles par année
medailles_par_annee = df['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.plot(medailles_par_annee.index, medailles_par_annee.values, marker='o', linestyle='-')
plt.xlabel('Année')
plt.ylabel('Nombre de médailles')
plt.title('Nombre de médailles par année aux Jeux olympiques')
plt.grid(True)
#Définir les étiquettes de l'axe x pour n'afficher que les multiples de 4
plt.xticks([year for year in medailles_par_annee.index if year % 4 == 0])
plt.show()

#Visualisation du nombre de médailles par pays
medailles_par_pays = df['NOC'].value_counts().sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
plt.barh(medailles_par_pays.index, medailles_par_pays.values)
plt.xlabel('Nombre de médailles')
plt.ylabel('Pays')
plt.title('Top 10 des pays avec le plus grand nombre de médailles depuis 1996')
plt.gca().invert_yaxis()  # Inverser l'ordre des pays pour afficher le plus haut en haut
plt.show()

#Visualisation des médailles par sport
medailles_par_sport = df['Sport'].value_counts().sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
plt.barh(medailles_par_sport.index, medailles_par_sport.values)
plt.xlabel('Nombre de médailles')
plt.ylabel('Sport')
plt.title('Top 10 des sports avec le plus grand nombre de médailles')
plt.gca().invert_yaxis()  # Inverser l'ordre des sports pour afficher le plus haut en haut
plt.show()

#On regarde le nb de non médailles vs médailles par Année de competition
df['Year'].value_counts().sort_index()
df['Year'].value_counts().sort_index().loc[1992:2016] #Ca fait deja 30 735 observations
df[(df['Year']>= 1992) & (df['Medal'].notna())] #Ca fait deja 30 735 observations

#Test, on doit ajouter le nb total de participant du dessus
df_filtered = df[df['Medal'].notna()]
medailles_par_annee = df_filtered['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.plot(medailles_par_annee.index, medailles_par_annee.values, marker='o', linestyle='-')
plt.xlabel('Année')
plt.ylabel('Nombre de médailles')
plt.title('Nombre de médailles par année aux Jeux olympiques')
plt.grid(True)
plt.xticks([year for year in medailles_par_annee.index if year % 4 == 0])  # Pour des années multiples de 4 sur l'axe x
plt.show()

#On regarde le nb de médailles par sport
df['Sport'].value_counts().sort_index()
df['Sport'].value_counts().sort_index().sum() #Ca fait deja 30735 observations
#df[(df['Year']>= 1992) & (df['Medal'].notna())] #Ca fait deja 30735 observations

#Par exemple ici on prend les medailles en Judo
judo_data = df[df['Sport'] == 'Judo']
judo_data


#%%traitement
#On copie la BDD
data=df.copy()

#On retire les doublons
data.drop_duplicates(inplace=True)

#On regarde les valeurs manquantes
df.isnull().sum()

#Exemple pour l'âge le plus vieux
data['Age']
data['Age'].describe()
data[data['Age']==88]['Sport']

#
sns.kdeplot(data=data,x="Age",shade=True)


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
#%%
df_carte = df.groupby(['Year', 'Team'])['Medal'].count().reset_index()
df_carte










