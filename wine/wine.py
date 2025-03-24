import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("wine/wine.data",header= None )

data.columns = [
    "classe", "alcool", "acido_malico", "cinzas", "alcalinidade_cinzas",
    "magnesio", "fenois_totais", "flavonoides", "fenois_nao_flavonoides",
    "proantocianinas", "intensidade_cor", "matiz",
    "OD280_OD315", "prolina"
]

x = data.drop("classe", axis=1)
y = data["classe"]

x_treino, x_prova, y_treino, y_prova = train_test_split(x, y, test_size=0.2, random_state=19)

#                               2 - 3 - 5         uniform - distance
modelo = KNeighborsClassifier(n_neighbors=2, weights='distance')

modelo.fit(x_treino, y_treino)

print(classification_report(y_prova,modelo.predict(x_prova)))
print(confusion_matrix(y_prova, modelo.predict(x_prova)))