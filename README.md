# Introdução

O projeto a seguir visa rotular vinhos utilizando aprendizado não supervisionado de agrupamento. E para isso, o KMeans foi escolhido para
realizar tal agrupamento. O dataset possui dados que são resultados de uma análise química de vinhos cultivados na mesma região da Itália, 
mas derivados de três cultivares diferentes. A análise determinou as quantidades de 13 constituintes encontrados em cada um dos três tipos de vinhos,
sendo eles definidos abaixo:

    - Alcohol : Álcool
    - Malic acid : Ácido Málico
    - Ash : Cinzas
    - Alcalinity of ash : Alcalinidade das Cinzas
    - Magnesium : Magnésio
    - Total phenols : Fenóis Totais
    - Flavanoids : Flavonóides
    - Nonflavanoid phenols : Fenóis Não Flavonóides
    - Proanthocyanins : Proantocianinas
    - Color intensity : Intensidade da Cor do Vinho
    - Hue : Matiz do Vinho
    - OD280/OD315 of diluted wines : Método de Determinação da Concentração de Proteínas Em Vinhos
    - Proline : Prolina
    
# Etapa de Machine Learning

Abaixo será descrito o passo a passo para realizar o estudo deste dataset com o algortimo KMeans. 

Primeiro, carregamos todas as bibliotecas necessárias:

```
# Manipulação de dados
import pandas as pd

# Visualização de dados
import seaborn as sns
import matplotlib.pyplot as plt

# Normalização dos dados
from sklearn.preprocessing import MinMaxScaler

# Redução de dimensionalidade
from sklearn.decomposition import PCA

# Algoritmo KMeans
from sklearn.cluster import KMeans

# Silhouette Score
from sklearn.metrics import silhouette_score
```

Carregando o dataset com o pandas:

```
df = pd.read_csv('./dataset/wine-clustering.csv')
```

Aqui observamos que o dataset possui o tipo de variáveis corretas ao que lhe foi atribuído:

```
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 178 entries, 0 to 177
Data columns (total 13 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   Alcohol               178 non-null    float64
 1   Malic_Acid            178 non-null    float64
 2   Ash                   178 non-null    float64
 3   Ash_Alcanity          178 non-null    float64
 4   Magnesium             178 non-null    int64  
 5   Total_Phenols         178 non-null    float64
 6   Flavanoids            178 non-null    float64
 7   Nonflavanoid_Phenols  178 non-null    float64
 8   Proanthocyanins       178 non-null    float64
 9   Color_Intensity       178 non-null    float64
 10  Hue                   178 non-null    float64
 11  OD280                 178 non-null    float64
 12  Proline               178 non-null    int64  
dtypes: float64(11), int64(2)
memory usage: 18.2 KB
```

Nao há valores missing ou NaN:

```
df.isnull().sum()

Alcohol                 0
Malic_Acid              0
Ash                     0
Ash_Alcanity            0
Magnesium               0
Total_Phenols           0
Flavanoids              0
Nonflavanoid_Phenols    0
Proanthocyanins         0
Color_Intensity         0
Hue                     0
OD280                   0
Proline                 0
dtype: int64
```

Normalizando os dados com o MinMaxScaler, a fim de manter os valores entre 0 e 1:

```
norm = MinMaxScaler()
df_scaler = norm.fit_transform(df)
```

Agora estamos prontos para realizar o agrupamento com KMeans, mas antes, é necessário passar o número correto de clusters (k) para o algoritmo,
o que é um pouco estranho, já que alguns poderiam esperar que o agrupamento fornecesse tal valor de forma independente. Porém, não é o caso. 
Para descobrir o valor k, utiliza-se o método Elbow, que compara duas grandezas, a inércia (que é a medida do quão bem um conjunto de dados 
é agrupado pelo K-Means) e o valor de clusters (k). O valor de k ótimo é obitdo plotando tal gráfico e obtendo o ponto "cotovelo". Abaixo é mostrado tal passo:


```
# Descobrindo o numero de clusters (k) com o ELBOW METHOD.

inertia = []
for k in range(2,11,1):
    kmeans = KMeans(n_clusters=k, random_state=99)
    kmeans.fit(df_scaler)
    inertia.append(kmeans.inertia_)


# Plotagem do Gráfico

x = list(range(2,11,1))
fig = plt.figure(figsize=(12,6))
sns.lineplot(x=x, y=inertia, marker='o', markersize=10, color='black', lw=2, mfc='red')
plt.xlabel('Número de Clusters (K)', fontsize=15)
plt.ylabel('Inércia', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
```
O resultado é a Figura abaixo:

<div align="center">
  <img src="https://user-images.githubusercontent.com/102380417/188332249-8a20cc7f-abb5-4a8f-b788-991791b69c79.png" width="700px" />
</div>

Deste gráfico, poderia-se escolher o valor de k = 3 para a realização do agrupamento, porém, pode-se utilizar o silhouette score (Outra medida
do quão bom a clusterização é para um dado k) em função do número de clusters. O mesmo procedimento realizado acima, é feito para o silhouette score:

```
sil = []
for k in range(2,11,1):
    kmeans = KMeans(n_clusters=k, random_state=99)
    kmeans.fit_predict(df_scaler)
    score = silhouette_score(df_scaler, kmeans.labels_)
    sil.append(score)
    

fig = plt.figure(figsize=(12,6))
sns.lineplot(x=x, y=sil, marker='o', markersize=10, color='black', lw=2, mfc='red')
plt.ylabel('Silhouette Score', fontsize=15)
plt.xlabel('Número de Clusters (K)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
```

O gráfico mostra que o valor de k = 2 e k = 3 são muito parecidos, dado não muito visível para o gráfico da inércia X k. Mas mesmo sendo parecidos,
o valor para k = 3 possui um silhouette score maior do que para k = 2:


<div align="center">
  <img src="https://user-images.githubusercontent.com/102380417/188332720-748947a4-2c7b-4d12-bf35-10e6e29a9830.png" width="700px" />
</div>

Agora sim pode-se realizar o treinamento do KMeans utilizando o número de cluster correto (k=3). Além disso, já adiciona ao Dataframe original os valores
dos labels armazenados na variável y_kmeans, lembrando que os labels adquiridos são referentes aos números de cada cluster.

```
# Treinamento dos algoritmo

kmeans1 = KMeans(n_clusters=3, random_state=99)
y_kmeans = kmeans1.fit_predict(df_scaler)

# Adicioandno labels ao DataFrame original

agroup = pd.DataFrame(y_kmeans, columns=['label'])
df_final = pd.concat([df, agroup], axis=1)
```

Das 178 entradas do DataFrame original, 55 pertecem ao cluster 0, 63 pertencem ao cluster 1 e 60 pertecem ao cluster 2. 

```
df_final['label'].value_counts()

1    63
2    60
0    55
Name: label, dtype: int64
```

Deixar rotulado apenas como
o número do cluster não é tão favorável assim, deve-se analisar um pouco mais os dados para poder atribuir labels corretos aos clusters, dado que 
a descrição do dataset nos diz que são vinhos de tres cultivos diferentes, então o nosso trabalho seria tentar atribuir cada cluster a um dado cultivo.
Porém, essa tarefa não será realizada.

Uma outra coisa que pode ser feita é a visualização de fato dos clusters. Existem formas de fazer a plotagem dos clusters para dados com dimensões maiores do que 2, mas para fins didáticos, realizarei a redução de dimensionalidade utilziando o PCA, que também é um algoritmo de aprendizado não supervisionado.

```
# Realizando a cópia do dataset original
df_pca = df.copy()

# Reduzindo a dimensionalidade do dataset para 2 para a plotagem do gráfico abaixo
pca = PCA(n_components=2)
cpca = pca.fit_transform(df_pca)

# Novamente treinando o algoritmo, mas agora com o dataset com dimensão reduzida
kmeans2 = KMeans(n_clusters=3, random_state=99)
kmeans2.fit(cpca)

# Adquirindo os labels dos centroids e dos pontos em si
centroids = kmeans2.cluster_centers_
labels = kmeans2.labels_

# Plotagem do gráfico
plt.figure(figsize=(12,6))
sns.set_theme(style='whitegrid')
sns.scatterplot(x=cpca[:,0], y=cpca[:,1], hue=clas, palette=["b", "g", "r"], s=50)
sns.scatterplot(x=centroids[:,0], y=centroids[:,1], s = 200)
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/102380417/188333514-11215958-1320-40fc-b3f4-b2c46bff5c0d.png" width="700px" />
</div>
