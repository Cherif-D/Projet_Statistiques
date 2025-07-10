import os
from pathlib import Path
import pandas as pd

from scripts import EDA, modelisation

def afficher_resultats_lineaire(linear_results):
    """
    Affiche les métriques, hypothèses et chemins des plots du modèle linéaire.
    """
    if 'erreur' in linear_results:
        print(f"Erreur lors de l'évaluation du modèle linéaire : {linear_results['erreur']}")
        return
    
    print("Métriques du modèle linéaire :")
    for metric_name, metric_value in linear_results['metrics'].items():
        print(f"  - {metric_name} : {metric_value:.4f}")

    print("\nHypothèses du modèle linéaire :")
    for hyp_name, hyp_data in linear_results['hypotheses'].items():
        print(f"  - {hyp_name} : {hyp_data['conclusion']}")
        if 'p_value' in hyp_data:
            print(f"    (p-value = {hyp_data['p_value']:.4f})")

    print("\nGraphiques des hypothèses générés :")
    for hyp_name, hyp_data in linear_results['hypotheses'].items():
        if 'plot' in hyp_data:
            plot_path = Path(hyp_data['plot'])
            if plot_path.exists():
                print(f"  - {hyp_name} : {plot_path.resolve()}")
            else:
                print(f"  - {hyp_name} : fichier {plot_path} non trouvé.")

def main():
    print("\n" + "="*80)
    print(" DÉBUT DU PIPELINE COMPLET : EDA + MODÉLISATION ".center(80, "="))
    print("="*80 + "\n")

    dossier_data = "data"

    print(">>> Étape 1 : Exécution de l'EDA complète\n")
    EDA.run_single_file_eda_integrated(dossier_data)

    fichier_nettoye = os.path.join(dossier_data, "air_quality_cleaned.csv")
    if not os.path.exists(fichier_nettoye):
        print(f"\n[ERREUR] Fichier nettoyé '{fichier_nettoye}' non trouvé, arrêt.")
        return

    print("\n>>> Étape 2 : Chargement des données nettoyées pour modélisation\n")
    df_clean = modelisation.charger_donnees_nettoyées(fichier_nettoye)
    print(f"Shape des données nettoyées : {df_clean.shape}\n")

    print(">>> Étape 3 : Préparation des variables explicatives (X) et cible (y)\n")
    X, y = modelisation.preparer_modelisation(df_clean)
    print(f"Shape de X : {X.shape}")
    print(f"Shape de y : {y.shape}\n")

    print(">>> Étape 4 : Modèle de régression linéaire (OLS)\n")
    linear_results = modelisation.evaluer_modele_lineaire(X, y, "Air_Quality")

    afficher_resultats_lineaire(linear_results)

    print("\n>>> Étape 5 : Modèles avancés (Random Forest, XGBoost, Decision Tree)\n")
    advanced_results = modelisation.evaluer_modeles_avances(X, y, "Air_Quality")

    if 'erreur_globale' not in advanced_results:
        for model_name, res in advanced_results.items():
            if 'erreur' not in res:
                print(f"\nRésultats pour {model_name}:")
                print(f"  Métriques :")
                for metric_name, metric_val in res['metrics'].items():
                    print(f"    - {metric_name} : {metric_val:.4f}")
                print("  Importance des variables (Top 3) :")
                for imp in res['importance'][:3]:
                    print(f"    - {imp['Feature']}: {imp['Importance']:.4f}")
            else:
                print(f"\nImpossible d'évaluer le modèle {model_name} : {res['erreur']}")
    else:
        print(f"Erreur lors de l'évaluation des modèles avancés : {advanced_results['erreur_globale']}")

    print("\n" + "="*80)
    print(" FIN DU PIPELINE COMPLET : EDA + MODÉLISATION ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
