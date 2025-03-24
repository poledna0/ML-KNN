import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("iris/iris.data")

data.columns = ["comp_sepoaneo", "argura_sepala", "comp_petala", "larg_petala", "especie"]

x = data.drop("especie", axis=1)
y = data["especie"]

x_treino, x_prova, y_treino, y_prova = train_test_split(x, y, test_size=0.2, random_state=19)

modelo = KNeighborsClassifier(n_neighbors=3, weights='distance')
modelo.fit(x_treino, y_treino)

print(classification_report(y_prova,modelo.predict(x_prova)))
print(confusion_matrix(y_prova, modelo.predict(x_prova)))