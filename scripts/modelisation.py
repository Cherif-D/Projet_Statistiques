import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Doit être appelé avant d'importer pyplot
import sys # Import pour la redirection de la sortie
import datetime # Pour horodater le fichier de rapport

# Imports spécifiques à la modélisation
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats# Pour le test de Shapiro-Wilk
from statsmodels.stats.api import het_goldfeldquandt
from statsmodels.graphics.tsaplots import plot_acf

# --- Classe pour rediriger la sortie de la console vers un fichier ---
class ReportWriter:
    """
    Redirige la sortie standard (stdout) vers un fichier et vers la console.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a") # Ouvre le fichier en mode ajout
        sys.stdout = self # Redirige stdout vers cette instance

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Cette méthode est nécessaire pour les flux de sortie
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Rétablit la sortie standard et ferme le fichier de log
        sys.stdout = self.terminal
        self.log.close()

# --- Fonction pour charger les données nettoyées ---
def charger_donnees_nettoyées(chemin_fichier_nettoye: str) -> pd.DataFrame:
    """
    Charge les données nettoyées depuis un fichier CSV.

    Args:
        chemin_fichier_nettoye (str): Le chemin complet vers le fichier CSV nettoyé.

    Returns:
        pd.DataFrame: Le DataFrame des données nettoyées.
    """
    if not os.path.exists(chemin_fichier_nettoye):
        raise FileNotFoundError(f"Le fichier de données nettoyées n'a pas été trouvé à : {chemin_fichier_nettoye}")
    
    print(f"Chargement des données nettoyées depuis : {chemin_fichier_nettoye}")
    df_cleaned = pd.read_csv(chemin_fichier_nettoye)
    print("Données nettoyées chargées avec succès.")
    
    # Assurer que la colonne 'Date' est au bon format si elle est présente et nécessaire pour des traitements ultérieurs
    if 'Date' in df_cleaned.columns:
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')
        # Si vous n'avez plus besoin de la colonne 'Date' après le nettoyage, vous pouvez la supprimer ici.
        # df_cleaned.drop(columns=['Date'], inplace=True)
    
    return df_cleaned


# --- Fonctions de Modélisation ---

def preparer_modelisation(df):
    """
    Prépare les données pour la modélisation en sélectionnant les features et la cible.
    Les features 'sin_mois' et 'cos_mois' sont incluses pour la saisonnalité.
    """
    features = ['NO2', 'O3', 'PM2.5',] # 'CO', 'PM10'
    
    # Assurez-vous que les colonnes 'annee', 'mois', 'jour_semaine' sont numériques ou gérées
    # Si 'mois' et 'jour_semaine' sont utiles pour la modélisation, ajoutez-les ici.
    # Si vous voulez encoder la saisonnalité (sin/cos pour le mois), c'est ici que vous le feriez.
    # Par exemple :
    # if 'mois' in df.columns:
    #     df['sin_mois'] = np.sin(2 * np.pi * df['mois']/12)
    #     df['cos_mois'] = np.cos(2 * np.pi * df['mois']/12)
    #     features.extend(['sin_mois', 'cos_mois'])

    features = [f for f in features if f in df.columns]
    
    if 'AQI' not in df.columns:
        raise ValueError("La colonne 'AQI' (cible) est manquante dans le DataFrame.")
    
    X = df[features]
    y = df['AQI']
    
    if X.empty or y.empty:
        raise ValueError("Les données X ou y sont vides après la sélection des caractéristiques.")
    
    return X, y

def verifier_hypotheses(model, X, y, nom_analyse):
    """
    Vérifie les hypothèses de la régression linéaire :
    - Linéarité
    - Normalité des résidus
    - Multicolinéarité
    - Homoscédasticité (test de Goldfeld-Quandt)
    - Autocorrélation des résidus
    - Points influents (distance de Cook)
    """
    results = {}
    plots_dir = Path('output/plots/hypotheses')
    plots_dir.mkdir(exist_ok=True, parents=True)

    residuals = model.resid

    # 1. Linéarité
    plt.figure(figsize=(10, 6))
    plt.scatter(model.predict(X), residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Linéarité (Résidus vs. Prédits) - {nom_analyse}')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Résidus')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    linearite_path = plots_dir / f"{nom_analyse}_linearite.png"
    plt.savefig(linearite_path, dpi=300)
    plt.close()
    results['linearite'] = {
        'plot': str(linearite_path),
        'conclusion': "Inspection visuelle requise. Idéalement, les résidus sont dispersés aléatoirement autour de zéro."
    }

    # 2. Normalité des résidus
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'QQ-Plot des Résidus - {nom_analyse}')
    plt.xlabel('Quantiles Théoriques')
    plt.ylabel('Quantiles des Résidus')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    normalite_path = plots_dir / f"{nom_analyse}_qqplot.png"
    plt.savefig(normalite_path, dpi=300)
    plt.close()

    if len(residuals) > 5000:
        stat_norm, p_value_norm = stats.normaltest(residuals)
        test_name = "D'Agostino-Pearson"
    else:
        stat_norm, p_value_norm = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk"

    conclusion_norm = f"Test de {test_name}: "
    if p_value_norm < 0.05:
        conclusion_norm += "Les résidus ne suivent PAS une distribution normale (p < 0.05)."
    else:
        conclusion_norm += "Les résidus suivent une distribution normale (p ≥ 0.05)."

    results['normalite'] = {
        'plot': str(normalite_path),
        'p_value': p_value_norm,
        'conclusion': conclusion_norm
    }

    # 3. Multicolinéarité (VIF)
    if 'const' in X.columns:
        X_no_const = X.drop(columns='const')
    else:
        X_no_const = X

    vifs_data = []
    if not X_no_const.empty and X_no_const.shape[1] > 0:
        for i in range(X_no_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_no_const.values, i)
            except np.linalg.LinAlgError:
                vif_val = np.inf
            vifs_data.append({'Variable': X_no_const.columns[i], 'VIF': vif_val})
        vifs_df = pd.DataFrame(vifs_data)
    else:
        vifs_df = pd.DataFrame(columns=['Variable', 'VIF'])

    plt.figure(figsize=(12, 6))
    if not vifs_df.empty:
        sns.barplot(x='VIF', y='Variable', data=vifs_df.sort_values('VIF', ascending=False))
    else:
        plt.text(0.5, 0.5, "Pas de variables pour calculer le VIF", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.axvline(x=5, color='orange', linestyle='--', label='VIF > 5 (potentielle multicolinéarité)')
    plt.axvline(x=10, color='r', linestyle='--', label='VIF > 10 (forte multicolinéarité)')
    plt.title(f'Variance Inflation Factor (VIF) - {nom_analyse}')
    plt.xlabel('VIF')
    plt.ylabel('Variable')
    plt.legend()
    plt.tight_layout()
    vif_path = plots_dir / f"{nom_analyse}_vif.png"
    plt.savefig(vif_path, dpi=300)
    plt.close()

    conclusion_multi = "Pas de multicolinéarité significative détectée (tous VIF < 5)."
    if not vifs_df.empty and (vifs_df['VIF'] > 5).any():
        if (vifs_df['VIF'] > 10).any():
            conclusion_multi = "Forte multicolinéarité détectée (certains VIF > 10). À investiguer."
        else:
            conclusion_multi = "Multicolinéarité potentielle détectée (certains VIF > 5). À surveiller."

    results['multicolinearite'] = {
        'VIF': vifs_df.set_index('Variable')['VIF'].to_dict() if not vifs_df.empty else {},
        'conclusion': conclusion_multi,
        'plot': str(vif_path)
    }

    # 4. Homoscédasticité (Goldfeld-Quandt)
    f_stat, p_val, _ = het_goldfeldquandt(residuals, X)
    plt.figure(figsize=(10, 6))
    plt.scatter(model.predict(X), residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Homoscedasticité - Test de Goldfeld-Quandt (p = {p_val:.3f})')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Résidus')
    plt.tight_layout()
    homoscedasticite_path = plots_dir / f"{nom_analyse}_homoscedasticite.png"
    plt.savefig(homoscedasticite_path, dpi=300)
    plt.close()
    results['homoscedasticite'] = {
        'p_value': p_val,
        'conclusion': "Homoscedasticité violée (p < 0.05)." if p_val < 0.05 else "Pas de preuve de violation d’homoscédasticité.",
        'plot': str(homoscedasticite_path)
    }

    # 5. Autocorrélation des résidus
    plt.figure(figsize=(10, 6))
    plot_acf(residuals, lags=40)
    plt.grid(True)
    plt.title(f'Autocorrélation des résidus - {nom_analyse}')
    plt.tight_layout()
    autocorr_path = plots_dir / f"{nom_analyse}_residus_autocorrelation.png"
    plt.savefig(autocorr_path, dpi=300)
    plt.close()
    results['autocorrelation'] = {
        'plot': str(autocorr_path),
        'conclusion': "Vérifier visuellement l’absence de structure dans l’ACF (bruit blanc attendu)."
    }

    # 6. Points influents (Distance de Cook)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    plt.figure(figsize=(10, 6))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
    plt.axhline(4 / len(cooks_d), color='r', linestyle='--', label='Seuil 4/n')
    plt.title(f"Distance de Cook - {nom_analyse}")
    plt.xlabel("Observation")
    plt.ylabel("Distance de Cook")
    plt.legend()
    plt.tight_layout()
    cooks_path = plots_dir / f"{nom_analyse}_cooks_distance.png"
    plt.savefig(cooks_path, dpi=300)
    plt.close()
    results['points_influents'] = {
        'plot': str(cooks_path),
        'nb_points_influents': int(sum(cooks_d > (4 / len(cooks_d)))),
        'conclusion': "Présence de points influents détectée." if any(cooks_d > 4 / len(cooks_d)) else "Pas de points influents détectés."
    }

    return results


def evaluer_modele_lineaire(X, y, nom_analyse):
    """
    Évalue un modèle de régression linéaire OLS (Ordinary Least Squares)
    avec statsmodels, sauvegarde le résumé et vérifie les hypothèses.
    """
    try:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit(cov_type='HC3')
        
        summary_dir = Path('output/summaries')
        summary_dir.mkdir(exist_ok=True, parents=True)
        with open(summary_dir / f"{nom_analyse}_lineaire_summary.txt", "w") as f:
            f.write(str(model.summary()))
        
        hypotheses = verifier_hypotheses(model, X_with_const, y, nom_analyse)
        
        y_pred = model.predict(X_with_const)
        metrics = {
            'R2': model.rsquared,
            'R2_adj': model.rsquared_adj,
            'MSE': mean_squared_error(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred)
        }
        # Ajout du plot Prédictions vs Vraies Valeurs pour le modèle linéaire
        plots_dir = Path('output/plots')
        plt.figure(figsize=(8, 8))
        plt.scatter(y, y_pred, alpha=0.5) # y sont les vraies valeurs, y_pred les prédictions
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Ligne y=x
        plt.title(f'Prédictions vs Vraies Valeurs (Linéaire) - {nom_analyse}')
        plt.xlabel('Vraies Valeurs (AQI)')
        plt.ylabel('Prédictions (AQI)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(plots_dir / f'{nom_analyse}_linear_predictions_vs_true.png', dpi=300)
        plt.close()
        
        return {
            'model': model,
            'metrics': metrics,
            'hypotheses': hypotheses
        }
    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle linéaire pour {nom_analyse}: {str(e)}")
        return {'erreur': str(e)}

def evaluer_modeles_avances(X, y, nom_analyse):
    """
    Évalue les modèles de régression avancés (Random Forest, XGBoost, Decision Tree).
    Effectue un split train/test, une mise à l'échelle, et calcule les métriques
    ainsi que l'importance des variables.
    """
    resultats = {}
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        modeles = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }
        
        for name, model in modeles.items():
            try:
                print(f"   Entraînement du modèle {name} pour {nom_analyse}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                metrics = {
                    'R2': r2_score(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2_cv': np.mean(cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1))
                }
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                    importances = result.importances_mean
                
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                result_dir = Path('output/model_results')
                result_dir.mkdir(exist_ok=True, parents=True)
                importance_df.to_csv(result_dir / f"{nom_analyse}_{name}_importance.csv", index=False)
                
                plt.figure(figsize=(10, max(6, len(importance_df) * 0.5)))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f'Importance des variables - {name} - {nom_analyse}')
                plt.tight_layout()
                plt.savefig(f'output/plots/{nom_analyse}_{name}_importance.png', dpi=300)
                plt.close()
                # --- Ajout du plot Prédictions vs Vraies Valeurs pour les modèles avancés ---
                plt.figure(figsize=(8, 8))
                plt.scatter(y_test, y_pred, alpha=0.5)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # Ligne y=x
                
                plt.title(f'Prédictions vs Vraies Valeurs ({name}) - {nom_analyse}')
                plt.xlabel('Vraies Valeurs (AQI)')
                plt.ylabel('Prédictions (AQI)')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/plots/{nom_analyse}_{name}_predictions_vs_true.png', dpi=300)
                plt.close()
                # --- Fin de l'ajout du plot ---
                
                resultats[name] = {
                    'metrics': metrics,
                    'importance': importance_df.to_dict('records')
                }
                
            except Exception as e:
                print(f"   Erreur lors de l'évaluation du modèle {name} pour {nom_analyse}: {str(e)}")
                resultats[name] = {'erreur': str(e)}
        
    except Exception as e:
        print(f"Erreur générale dans evaluer_modeles_avances pour {nom_analyse}: {str(e)}")
        resultats = {'erreur_globale': str(e)}
        
    return resultats

# --- Point d'entrée du script de Modélisation ---
if __name__ == "__main__":
    # Définir le chemin vers le fichier de données nettoyées
    # Cela doit correspondre au chemin et nom de fichier utilisés dans le script EDA
    chemin_fichier_nettoye = 'data/air_quality_cleaned.csv' 

    # --- Configuration du rapport de sortie ---
    report_dir = Path('output/reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = report_dir / f"rapport_modelisation_{timestamp}.txt"
    
    # Redirige la sortie standard
    original_stdout = sys.stdout # Sauvegarde la sortie standard originale
    sys.stdout = ReportWriter(report_filename)

    try:
        print("\n" + "=" * 70)
        print(f" Lancement du processus de Modélisation - {timestamp} ".center(70, '='))
        print("=" * 70)
        
        # Charger les données nettoyées
        df_cleaned = charger_donnees_nettoyées(chemin_fichier_nettoye)
        
        # Préparer les données pour la modélisation
        X, y = preparer_modelisation(df_cleaned)
        
        print("\n" + "=" * 70)
        print(" Lancement de la Modélisation ".center(70, '='))
        print("=" * 70)
        
        # Évaluation du modèle linéaire
        print("\n" + "--- Évaluation du Modèle Linéaire (OLS) ---".center(70, '-'))
        linear_results = evaluer_modele_lineaire(X, y, "Air_Quality_Model_Linear")
        if 'erreur' not in linear_results:
            print(f"Métriques Modèle Linéaire: {linear_results['metrics']}")
            print("Hypothèses du Modèle Linéaire:")
            for k, v in linear_results['hypotheses'].items():
                print(f"  - {k}: {v['conclusion']}")
                if 'p_value' in v:
                    print(f"    (p-value: {v['p_value']:.4f})")
        else:
            print(f"Impossible d'évaluer le modèle linéaire: {linear_results['erreur']}")

        # Évaluation des modèles avancés
        print("\n" + "--- Évaluation des Modèles Avancés ---".center(70, '-'))
        advanced_results = evaluer_modeles_avances(X, y, "Air_Quality_Model_Advanced")
        
        if 'erreur_globale' not in advanced_results:
            for model_name, res in advanced_results.items():
                if 'erreur' not in res:
                    print(f"\nRésultats pour {model_name}:")
                    print(f"  Métriques: {res['metrics']}")
                    print(f"  Importance des variables (Top 3):")
                    # Afficher les 3 premières variables les plus importantes
                    for imp in res['importance'][:3]:
                        print(f"    - {imp['Feature']}: {imp['Importance']:.4f}")
                else:
                    print(f"\nImpossible d'évaluer le modèle {model_name}: {res['erreur']}")
        else:
            print(f"Impossible d'évaluer les modèles avancés: {advanced_results['erreur_globale']}")

        print("\n" + "=" * 70)
        print(" Modélisation terminée. ".center(70, '='))
        print(" Les résumés et plots sont sauvegardés dans 'output/summaries' et 'output/plots'. ".center(70, '='))
        print(f" Le rapport complet est disponible dans : {report_filename} ".center(70, '='))
        print("=" * 70)

    except Exception as e:
        print(f"\nErreur critique lors de l'exécution du script de modélisation: {str(e)}")
    finally:
        # Rétablit la sortie standard à la fin, même en cas d'erreur
        if isinstance(sys.stdout, ReportWriter):
            sys.stdout.close()
            sys.stdout = original_stdout # Assurez-vous de restaurer l'original
            