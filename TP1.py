import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Chargement du fichier CSV
df = pd.read_csv("rendement_mais.csv")

# Affichage du dataframe pour vérifier qu'il a été chargé correctement
print(df)

# 2.1 Mesures de tendance centrale

# Calcul de la moyenne, médiane, et mode du rendement
rendement = df['RENDEMENT_T_HA'].values

# Moyenne
moyenne = np.mean(rendement)

# Médiane
mediane = np.median(rendement)

# Mode
mode = stats.mode(rendement)

print(f"\nMesures de tendance centrale :")
print(f"Moyenne du rendement : {moyenne}")
print(f"Médiane du rendement : {mediane}")
print(f"Mode du rendement : {mode.mode} (Fréquence : {mode.count})")

# 2.2 Mesures de dispersion

# Calcul de l'écart-type, variance et étendue
ecart_type = np.std(rendement)
variance = np.var(rendement)
etendue = np.ptp(rendement)

print(f"\nMesures de dispersion :")
print(f"Écart-type du rendement : {ecart_type}")
print(f"Variance du rendement : {variance}")
print(f"Étendue du rendement : {etendue}")

# 2.3 Visualisation des données

# Données pour précipitations et température
precipitations = df['PRECIPITATIONS_MM'].values
temperature = df['TEMPERATURE_C'].values

# Paramétrage des graphes
plt.figure(figsize=(12, 8))

# Histogramme du rendement
plt.subplot(2, 2, 1)
plt.hist(rendement, bins=5, color='skyblue', edgecolor='black')
plt.title("Histogramme du Rendement")
plt.xlabel("Rendement (t/ha)")
plt.ylabel("Fréquence")

# Histogramme des précipitations
plt.subplot(2, 2, 2)
plt.hist(precipitations, bins=5, color='lightgreen', edgecolor='black')
plt.title("Histogramme des Précipitations")
plt.xlabel("Précipitations (mm)")
plt.ylabel("Fréquence")

# Histogramme de la température
plt.subplot(2, 2, 3)
plt.hist(temperature, bins=5, color='lightcoral', edgecolor='black')
plt.title("Histogramme de la Température")
plt.xlabel("Température (°C)")
plt.ylabel("Fréquence")

# Boxplot du rendement
plt.subplot(2, 2, 4)
sns.boxplot(data=rendement, color='lightblue')
plt.title("Boxplot du Rendement")

# Affichage des graphiques
plt.tight_layout()
plt.show()

# Boxplot des précipitations et températures
plt.figure(figsize=(8, 5))
sns.boxplot(data=[precipitations, temperature], palette="Set2")
plt.xticks([0, 1], ['Précipitations', 'Température'])
plt.title("Boxplot des Précipitations et Température")
plt.show()


# 3. Matrice de Corrélation

# Sélection des colonnes numériques pour la corrélation
numerical_columns = df[['SURFACE_HA', 'ENGRAIS_KG_HA', 'PRECIPITATIONS_MM', 'TEMPERATURE_C', 'RENDEMENT_T_HA']]

# Calcul de la matrice de corrélation
correlation_matrix = numerical_columns.corr()

# Affichage de la matrice de corrélation
print("\nMatrice de Corrélation :")
print(correlation_matrix)

# 4. Visualisation de la heatmap de la corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap de la Matrice de Corrélation")
plt.show()

# 5. Test ANOVA détaillé : Influence du type de sol sur le rendement

# Regrouper les rendements par type de sol
groupes = df.groupby('TYPE_SOL')['RENDEMENT_T_HA'].apply(list)

# 1. Moyenne totale du rendement
moyenne_totale = np.mean(df['RENDEMENT_T_HA'])

# 2. Moyenne par type de sol
moyennes_groupes = groupes.apply(np.mean)

# 3. Calcul des sommes des carrés (SSB et SSW)

# Calcul de SSB
SSB = sum([len(groupe) * (np.mean(groupe) - moyenne_totale)**2 for groupe in groupes])

# Calcul de SSW
SSW = sum([sum((groupe - np.mean(groupe))**2) for groupe in groupes])

# 4. Degrés de liberté
k = len(groupes)  # Nombre de groupes
N = len(df)  # Nombre total d'observations

df_between = k - 1
df_within = N - k

# 5. Calcul des variances MSB et MSW
MSB = SSB / df_between
MSW = SSW / df_within

# 6. Calcul de la statistique F
F = MSB / MSW

# 7. Calcul de la p-value
p_val = stats.f.sf(F, df_between, df_within)

# Afficher les résultats
print("\nTest ANOVA détaillé :")
print(f"Moyenne totale du rendement : {moyenne_totale}")
print(f"\nMoyennes par type de sol :\n{moyennes_groupes}")
print(f"\nSomme des Carrés entre les groupes (SSB) : {SSB}")
print(f"Somme des Carrés intra-groupe (SSW) : {SSW}")
print(f"\nDegrés de liberté entre les groupes (df_between) : {df_between}")
print(f"Degrés de liberté à l'intérieur des groupes (df_within) : {df_within}")
print(f"\nVariance inter-groupe (MSB) : {MSB}")
print(f"Variance intra-groupe (MSW) : {MSW}")
print(f"\nStatistique F : {F}")
print(f"P-value : {p_val}")

# 8. Interprétation du test
if p_val < 0.05:
    print("\nLe type de sol influence significativement le rendement (p < 0.05).")
else:
    print("\nLe type de sol n'influence pas significativement le rendement (p >= 0.05).")




# 9. Diviser les données en train (80%) et test (20%)
X = df[['SURFACE_HA', 'ENGRAIS_KG_HA', 'PRECIPITATIONS_MM', 'TEMPERATURE_C']]  # Variables explicatives
y = df['RENDEMENT_T_HA']  # Variable cible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Création du modèle : Régression Linéaire et Arbre de Décision
# Modèle 1: Régression Linéaire
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Modèle 2: Arbre de Décision
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# 11. Évaluation des modèles

# Prédictions sur l'ensemble de test
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)

# Calcul des métriques pour la régression linéaire
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Calcul des métriques pour l'arbre de décision
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

# Affichage des résultats d'évaluation
print("\nÉvaluation des modèles :")

# Régression Linéaire
print("\nRégression Linéaire :")
print(f"MAE (Erreur absolue moyenne) : {mae_lr}")
print(f"RMSE (Racine de l'erreur quadratique moyenne) : {rmse_lr}")
print(f"R² : {r2_lr}")

# Arbre de Décision
print("\nArbre de Décision :")
print(f"MAE (Erreur absolue moyenne) : {mae_dt}")
print(f"RMSE (Racine de l'erreur quadratique moyenne) : {rmse_dt}")
print(f"R² : {r2_dt}")
