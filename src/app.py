#from utils import db_connect
#engine = db_connect()

# your code here

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# 1. CARGA DE DATOS
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
try:
    df = pd.read_csv(url)
except:
    df = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv")

# 2. PREPROCESAMIENTO
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ENTRENAMIENTO CON LOS MEJORES HIPERPARÁMETROS (Grid Search)
# Usamos exactamente las reglas que descubriste
best_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_leaf=4,
    min_samples_split=2,
    random_state=42
)
best_model.fit(X_train, y_train)

# 4. GUARDAR EL MODELO
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("¡Árbol de decisión optimizado y guardado con éxito como 'decision_tree_model.pkl'!")
