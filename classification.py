import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Assurez-vous que cette ligne est bien présente et au début
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

# Pour gérer le déséquilibre de classes
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Pour l'interprétabilité des modèles (SHAP)
import shap

# Configuration des styles de graphique
plt.style.use('ggplot')
sns.set_palette("husl")

# Supprimer les warnings pour une meilleure lisibilité
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- 1. Chargement et préparation des données ---
def trouver_chemin_fichier():
    """Trouve le chemin du fichier CSV selon la structure actuelle"""
    projet_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(projet_dir, 'data')
    fichiers_csv = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if 'air_quality_cleaned.csv' in fichiers_csv:
        chemin = os.path.join(data_dir, 'air_quality_cleaned.csv')
        print(f"Fichier trouvé: {chemin}")
        return chemin
    print("\nFichiers CSV disponibles dans data/:")
    for i, f in enumerate(fichiers_csv, 1):
        print(f"{i}. {f}")
    choix = input("\nEntrez le numéro du fichier à utiliser (ou 0 pour annuler): ")
    try:
        choix = int(choix)
        if 1 <= choix <= len(fichiers_csv):
            return os.path.join(data_dir, fichiers_csv[choix-1])
    except ValueError:
        pass
    print("\nERREUR: Aucun fichier sélectionné")
    return None

def charger_et_preparer_donnees(file_path=None, seuil_mauvais=50):
    """Charge et prépare les données de qualité de l'air."""
    if file_path is None:
        file_path = trouver_chemin_fichier()
        if file_path is None:
            return None
    print(f"\n=== CHARGEMENT DES DONNÉES ===")
    print(f"Chargement depuis: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Fichier chargé. Dimensions: {df.shape}")
    except Exception as e:
        print(f"\nERREUR de chargement: {str(e)}")
        return None
    print("\nTraitement des valeurs manquantes...")
    df = df.fillna(df.mean(numeric_only=True))
    if 'AQI' not in df.columns:
        print("ERREUR: Colonne 'AQI' manquante")
        return None
    print("\nCréation de la variable cible...")
    df['qualite_mauvaise'] = (df['AQI'] > seuil_mauvais).astype(int)
    df.drop('AQI', axis=1, inplace=True)
    print("\n=== APERÇU DES DONNÉES ===")
    print(df.head())
    print(f"\nDistribution des classes:")
    print(df['qualite_mauvaise'].value_counts(normalize=True))
    print("\n" + "="*80)
    print("\nTraitement des colonnes non-numériques...")
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            print("Colonne 'Date' convertie en datetime")
        except Exception as e:
            print(f"Erreur conversion date: {str(e)}")
            return None
    print("\nVérification des types de colonnes:")
    print(df.dtypes)
    return df

def preparer_donnees_pour_modelisation(df_input, is_spatial_analysis=False):
    """Prépare les données pour la modélisation."""
    df = df_input.copy()
    if 'Date' in df.columns:
        df['annee'] = df['Date'].dt.year
        df['mois'] = df['Date'].dt.month
        df['jour'] = df['Date'].dt.day
        df['heure'] = df['Date'].dt.hour
        df.drop('Date', axis=1, inplace=True)
    colonne_ville = 'City' if 'City' in df.columns else 'ville' if 'ville' in df.columns else None
    if colonne_ville:
        if is_spatial_analysis:
            df = df.drop(colonne_ville, axis=1) 
        else:
            df = pd.get_dummies(df, columns=[colonne_ville], drop_first=True, dtype=int)
    non_numeriques = df.select_dtypes(exclude=['number']).columns
    if len(non_numeriques) > 0:
        print(f"\nATTENTION: Colonnes non-numériques restantes - {list(non_numeriques)}")
        print("Conversion forcée en numérique...")
        for col in non_numeriques:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    X = df.drop('qualite_mauvaise', axis=1)
    y = df['qualite_mauvaise']
    return X, y

# --- 2. Modélisation et évaluation ---
def evaluer_classification(X, y):
    """Évalue plusieurs modèles de classification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', SMOTE(random_state=42)),
            ('model', LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'))
        ]),
        'Random Forest': ImbPipeline([
            ('sampler', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'XGBoost': ImbPipeline([
            ('sampler', SMOTE(random_state=42)),
            ('model', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1))
        ])
    }

    param_grids = {
        'Logistic Regression': {'model__C': [0.1, 1.0, 10.0]},
        'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]},
        'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
    }

    results = []

    for name, pipeline in models.items():
        print(f"\n{'='*50}\nModèle: {name}\n{'='*50}")
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grids[name],
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
        print(classification_report(y_test, y_pred))
        
        results.append({
            'model': name,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model_obj': best_model,
            # NE PLUS STOCKER X_test ICI POUR ÉVITER LES PROBLÈMES AVEC PANDAS LORS DE LA CONCATÉNATION
            # 'X_test': X_test 
        })

        # Visualisations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(results[-1]['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bonne', 'Mauvaise'], yticklabels=['Bonne', 'Mauvaise'], ax=axes[0])
        axes[0].set_title(f'Matrice de confusion - {name}')
        
        RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1])
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title(f'Courbe ROC - {name}')
        
        plt.tight_layout()
        plt.show()

    # MODIFICATION : Retourner X_test et y_test séparément de la DataFrame des résultats
    return pd.DataFrame(results), X_test, y_test

# --- 3. Analyse SHAP ---
def analyser_importances(results_df, X_data_for_shap, title_prefix=""): 
    """
    Analyse l'importance des variables avec SHAP pour les modèles passés.
    X_data_for_shap est le X_test pertinent pour cette analyse (global ou spécifique à la ville).
    """
    for _, row in results_df.iterrows():
        model_name = row['model']
        # On ne veut faire le SHAP que pour la Régression Logistique
        if model_name != 'Logistic Regression':
            continue 
        
        model = row['model_obj'].named_steps['model']
        
        print(f"\n{'='*50}\nImportance SHAP - {title_prefix}{model_name}\n{'='*50}")

        # Pour Logistic Regression, utiliser LinearExplainer
        scaler = row['model_obj'].named_steps['scaler']
        X_scaled = scaler.transform(X_data_for_shap) 
        
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data_for_shap, plot_type="bar", show=False) 
        plt.title(f'Importance SHAP - {title_prefix}{model_name}')
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data_for_shap, show=False) 
        plt.title(f'Valeurs SHAP - {title_prefix}{model_name}')
        plt.show()

# --- 4. Analyse par ville ---
def classification_par_ville(df_full):
    """Analyse séparée par ville."""
    if 'City' not in df_full.columns:
        print("Colonne 'City' absente.")
        return None

    villes = df_full['City'].unique()
    all_results = []

    for ville in villes:
        print(f"\n{'='*50}\nCity: {ville}\n{'='*50}")
        df_ville = df_full[df_full['City'] == ville].copy()
        
        X_city_full, y_city_full = preparer_donnees_pour_modelisation(df_ville, is_spatial_analysis=True)
        
        # Vérifier si la ville a assez de données et des classes différentes pour être modélisée
        if len(y_city_full.unique()) < 2 or X_city_full.shape[0] < 10: 
            print(f"Données insuffisantes ou non variables pour {ville}. Ignoré.")
            continue
            
        # MODIFICATION : evaluer_classification renvoie maintenant trois valeurs
        results_for_city_df, X_test_city, y_test_city = evaluer_classification(X_city_full, y_city_full)
        
        # --- DÉBUT DU BLOC D'INTÉGRATION SHAP POUR CHAQUE VILLE (Logistic Regression) ---
        # Itérer sur les résultats pour trouver le modèle de Régression Logistique
        for index, row_model in results_for_city_df.iterrows(): # Itérer sur la DataFrame des résultats
            if row_model['model'] == 'Logistic Regression': # Nous ne voulons que la Régression Logistique
                # Appel de la fonction d'analyse SHAP
                # On passe le X_test_city qui a été retourné séparément par evaluer_classification
                analyser_importances(pd.DataFrame([row_model]), X_test_city, title_prefix=f'Ville {ville} - ')
                break # Une fois la Régression Logistique traitée, sortir de cette boucle interne
        # --- FIN DU BLOC D'INTÉGRATION SHAP POUR CHAQUE VILLE ---

        # Ajouter les résultats de TOUS les modèles de cette ville à la liste globale
        results_for_city_df['ville'] = ville
        all_results.append(results_for_city_df)

    if all_results:
        return pd.concat(all_results)
    return None

# --- 5. Fonction principale ---
def run_full_analysis(seuil_mauvais=50, par_ville=True):
    """Lance l'analyse complète."""
    df = charger_et_preparer_donnees(seuil_mauvais=seuil_mauvais)
    if df is None:
        return

    # Analyse globale
    print("\nANALYSE GLOBALE")
    X_global_full, y_global_full = preparer_donnees_pour_modelisation(df)
    # MODIFICATION : Capturer les trois valeurs de retour de evaluer_classification
    global_results_df, X_test_global, y_test_global = evaluer_classification(X_global_full, y_global_full)
    # MODIFICATION : Passer le X_test_global explicitement à analyser_importances
    analyser_importances(global_results_df, X_test_global, title_prefix="Global - ")

    # Analyse par ville
    if par_ville:
        print("\nANALYSE PAR VILLE")
        ville_results = classification_par_ville(df)
        if ville_results is not None:
            print("\nRésultats par ville:")
            print(ville_results[ville_results['model'] == 'Logistic Regression'][['ville', 'model', 'f1', 'roc_auc']].round(3))


# --- Exécution ---
if __name__ == '__main__':
    print("\n" + "="*80)
    print(f"Dossier de travail: {os.getcwd()}")
    print("Structure trouvée:")
    try:
        print(f"data/ : {os.listdir('data')}")
    except FileNotFoundError:
        print("Le dossier 'data' n'existe pas dans le répertoire courant")
    
    run_full_analysis(seuil_mauvais=50, par_ville=True)