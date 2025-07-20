🌫️ Projet Statistiques – Analyse de la Qualité de l’Air
🎯 Objectif
Ce projet a pour but d’identifier les facteurs influençant la qualité de l’air (AQI) à partir de données environnementales. Deux approches complémentaires sont explorées :

Prédiction directe de l’AQI à l’aide de méthodes de régression.

Classification binaire de l’AQI (bon/mauvais) à partir des seuils définis dans data_bis, afin d’identifier les variables explicatives les plus influentes.

Le projet interroge également la stabilité des facteurs explicatifs :

Sont-ils les mêmes selon les villes ?

Sont-ils constants dans le temps ?

🧪 Méthodologie
Prétraitement : nettoyage des données, gestion des valeurs manquantes, encodage des variables.

Exploration des données : visualisations, corrélations, comparaisons multi-villes.

Modélisation :

Régression linéaire, Ridge, Lasso pour la prédiction de l’AQI.

Arbre de décision et Random Forest pour la classification binaire.

Analyse de l’importance des variables et interprétation des résultats.

Approche multi-échelle : une analyse comparative entre plusieurs villes et périodes temporelles.

📈 Résultats
Identification de polluants clés (comme PM2.5, PM10) fortement liés à l’AQI.

Les facteurs explicatifs varient selon les villes, ce qui reflète des dynamiques locales spécifiques.

Les modèles développés offrent des performances satisfaisantes (MSE, RMSE), confirmant la robustesse de l’approche.

👥 Contributeurs
Ce projet a été réalisé en collaboration par :

TREZISE Komi 
TREZISE Yao
CHERIF Mamadou
