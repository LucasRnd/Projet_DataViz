import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

# Import des données
df = pd.read_csv("C:/Users/Lucas/Downloads/archive/dataset_olympics.csv", sep=",", encoding="utf-8")

# Filtrer les Jeux olympiques d'été depuis 1994
df = df[(df['Year'] >= 1994) & (df['Year'] % 4 == 0)]

# Convertir les colonnes 'Height', 'Weight' et 'Age' en numérique (au cas où elles seraient initialement des chaînes)
df[['Height', 'Weight', 'Age']] = df[['Height', 'Weight', 'Age']].apply(pd.to_numeric, errors='coerce')

# Supprimer les lignes avec des valeurs manquantes dans les colonnes 'Height', 'Weight' et 'Age'
df = df.dropna(subset=['Height', 'Weight', 'Age', 'Sex'])

# Remplacer les valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df[['Age', 'Height', 'Weight']])

# Convertir les étiquettes de médaille en valeurs numériques
medal_mapping = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
df['Medal'] = df['Medal'].replace(medal_mapping).fillna(0)

# Séparer les caractéristiques (X) et la cible (y)
X = df[['Age', 'Height', 'Weight']]
y = df['Medal']

# Suréchantillonnage des classes avec SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Diviser l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser le modèle de régression logistique
logistic_model = LogisticRegression(random_state=42)

# Recherche par grille pour les meilleurs hyperparamètres
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle
best_logistic_model = grid_search.best_estimator_
best_logistic_model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle de régression logistique
joblib.dump(best_logistic_model, 'best_logistic_model.pkl')

# Initialiser le modèle basé sur les arbres de décision
rf_model = RandomForestClassifier(random_state=42)

# Recherche par grille pour les meilleurs hyperparamètres pour le modèle de forêt aléatoire
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20, 30]}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle de forêt aléatoire
best_rf_model = grid_search_rf.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle de forêt aléatoire
joblib.dump(best_rf_model, 'best_rf_model.pkl')

# Faire des prédictions sur l'ensemble de test
y_pred_logistic = best_logistic_model.predict(X_test_scaled)
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Évaluer les performances des modèles
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Afficher les performances des deux modèles
print("Régression Logistique:")
print(f'Accuracy: {accuracy_logistic}')
print('Classification Report:')
print(classification_rep_logistic)

print("\nForêt Aléatoire:")
print(f'Accuracy: {accuracy_rf}')
print('Classification Report:')
print(classification_rep_rf)
