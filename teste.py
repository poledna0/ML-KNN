import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("iris.data")

# renomeando as colunas
data.columns = ["comp_sepoaneo", "argura_sepala", "comp_petala", "larg_petala", "especie"]

#axis=0 - axis=1 - refere as linhas do DataFrame.

x = data.drop("especie", axis=1)
y = data["especie"]

#print(data.head())

#print("Var independentes (X):")
#print(x.head())

#print("")
#print("Var dependente (y):")
#print(y.head())

x_treino, x_prova, y_treino, y_prova = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_treino.iloc[0]) tira print do primeiro vetor na matriz 


# mostra o tamanho deles
#print("treino:", x_treino.shape, y_treino.shape)
#print("testar:", x_prova.shape, y_prova.shape)

modelo = KNeighborsClassifier(n_neighbors=3, weights='distance')

# esta sendo treinado
modelo.fit(x_treino, y_treino)


print(classification_report(y_prova,modelo.predict(x_prova)))
print(confusion_matrix(y_prova, modelo.predict(x_prova)))