# -*- coding: utf-8 -*-
#Titanic - Quem sobreviveu? - Kaggle
import pandas as pd
import numpy as np
import csv

# Classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Será criada uma nova característica (feature) a partir de cada nome.
# Parece haver uma relação entre o título que uma pessoa têm e se eles sobreviveram ou não.
# Os títulos vêm dos dados que devem ser usados para a atividade do Kaggle (estão em inglês).
# Logo, vamos transformar a coluna de nomes em uma coluna de números representando os títulos.
# Exemplo: Se não tiver título = -1, se tiver de "Master" = 0, se tiver de "Mr." = 1, etc...
# Até preencher todos os títulos com um valor numérico correspondente.

def parseName(name):
  out = -1
  names = ["Master.", "Mr.", "Dona.", "Miss.", "Mrs.", "Dr.", "Rev.", "Col.", "Ms.", "Capt.", "Mlle.", "Major.", "Mme."]
  for i, n in enumerate(names):
    if n in name:
      out = i
      break
  return out

# Os dados vindos do Kaggle são otimizados para funcionarem.
def cleanData(data):
  # Se faltarem os dados da tarifa, substitua-os com a média dessa classe.
  data.Fare = data.Fare.map(lambda x: np.nan if x==0 else x)
  classmeans = data.pivot_table('Fare', rows='Pclass', aggfunc='mean')
  data.Fare = data[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )

  # Transforma nomes em números representando títulos.
  data.Name = data.Name.map(lambda x: parseName(x))

  # Encoberta sexo em um valor numérico
  data.Sex = data.Sex.apply(lambda sex: 0 if sex == "male" else 1)

  return data


# Carrega os modelos de treino e de dados, otimizando-os no processo.
train = cleanData(pd.read_csv("train.csv"))
test = cleanData(pd.read_csv("test.csv"))

# Pega as 4 colunas mais importantes e divide o modelo de treinamento em
# características (X) e rótulos (y).
cols = ["Fare", "Pclass", "Sex", "Name"]
X = train[cols].values
y = train['Survived'].values

# Para usar um SVC, os dados precisam ser escalados entre [-1, 1] com uma média de 0.
# O Dimensionamento não vai afetar negativamente nenhuma outra classificação.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Cria os classificadores para cada classificação.
clf1 = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
clf2 = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
clf3 = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
clf4 = AdaBoostClassifier(n_estimators=500)
clf5 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
clf6 = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.2, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)

clfs = [clf1, clf2, clf3, clf4, clf5, clf6]

# Ajusta cada classificador baseado nos dados de treinamento.
for clf in clfs:
  clf.fit(X, y)

# Cria as características do conjunto de teste.
X = test[cols].values
X = scaler.transform(X)

# Para todos os 6 classificadores, prediz as saídas e salva as probabilidades de cada predição.
predictions = []

for clf in clfs:
  predictions.append(clf.predict_proba(X))

# Note que agora há seis conjuntos de predições em uma lista.  
# Passa uma Média através todas as listas para criar uma média de predição através de todos os classificadores.
# Essa maneira mesmo não sendo a mais eficiente realmente funciona aqui.
p = np.mean(predictions, axis=0)

# Há uma lista onde cada elemento é uma tupla de probabilidade verdadeira ou falsa.
# Tranformando esses valores em 0 ou 1 baseado no valor da propriedade de verdadeiro.
p = map(lambda x: 0 if x[0] >= 0.5 else 1, p)

# Tem-se uma predição para cada item de 0 ou 1.
# Só é preciso escrever o resultado em arquivo de formato .csv que o Kaggle aceita.
with open('predictions.csv', 'wb') as csvfile:
  w = csv.writer(csvfile)
  w.writerow(["PassengerId", "Survived"])

  for i in xrange(len(p)):
    w.writerow([test.PassengerId[i], p[i]])
