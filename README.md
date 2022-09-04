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
Para descobrir o valor k, utiliza-se o método Elbow, que compara duas grandezas, a inércia (mede o quão bem um conjunto de dados é agrupado pelo K-Means)
e o valor de clusters (k). O valor de k ótimo é obitdo plotando tal gráfico e obtendo o ponto "cotovelo". Abaixo é mostrado tal passo:


```
# descobrindo o numero de clusters (k) com o ELBOW METHOD.

inertia = []
for k in range(2,11,1):
    kmeans = KMeans(n_clusters=k, random_state=99)
    kmeans.fit(df_scaler)
    inertia.append(kmeans.inertia_)
```
