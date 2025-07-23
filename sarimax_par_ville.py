import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
import warnings
import pmdarima as pm

# Supprimer les avertissements de convergence qui peuvent être verbeux
warnings.filterwarnings("ignore")

# ==========================================
# 1. Chargement et préparation des données
# ==========================================
def load_data(path):
    """
    Charge le fichier CSV contenant les données de qualité de l'air,
    en s'assurant que les colonnes essentielles sont présentes et sans valeurs manquantes.
    """
    print(f"\n--- Étape 1: Chargement et préparation des données ---")
    print(f"Chargement du fichier: {path}")
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        print(f"Dimensions initiales du DataFrame: {df.shape}")
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{path}' est introuvable. Vérifiez le chemin.")
        exit() # Quitte le programme si le fichier n'est pas trouvé

    # Assurez-vous que les colonnes essentielles existent
    required_columns = ['AQI', 'Date', 'City']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"ERREUR: La colonne '{col}' est introuvable dans le fichier CSV. Veuillez vérifier son nom.")
    
    initial_rows = df.shape[0]
    df = df.dropna(subset=required_columns)
    rows_after_dropna = df.shape[0]
    print(f"Nombre de lignes avant suppression des NaN: {initial_rows}")
    # CORRECTION ICI: Changement de 'rows_dropna' à 'rows_after_dropna'
    print(f"Nombre de lignes après suppression des NaN: {rows_after_dropna} ({initial_rows - rows_after_dropna} lignes supprimées)")
    print("Chargement et nettoyage initial terminés.")
    return df

# ==========================================
# 2. K-Fold temporel avec SARIMAX
# (Fonction générique pour une série, qu'elle soit par ville ou globale)
# ==========================================
def fit_sarimax_kfold(y_series, n_splits=5, s_period=7, label="Série"):
    """
    Applique SARIMAX avec validation croisée temporelle (TimeSeriesSplit)
    et trouve les meilleurs ordres SARIMA avec auto_arima pour une série donnée.
    Retourne la moyenne des RMSE/MAE, les prévisions finales, la série traitée
    le modèle ajusté, et les ordres SARIMA optimaux.
    """
    print(f"  > Préparation des données pour le K-Fold ({label})...")
    
    # Gérer les NaN qui pourraient apparaître (si la série n'a pas été parfaitement préparée en amont)
    missing_dates_before_interp = y_series.isnull().sum()
    if missing_dates_before_interp > 0:
        print(f"  > {missing_dates_before_interp} valeurs manquantes sur l'indice temporel continu. Interpolation linéaire...")
        y_series = y_series.interpolate(method='linear')
        y_series = y_series.fillna(method='bfill').fillna(method='ffill')
        if y_series.isnull().sum() > 0:
            print("  AVERTISSEMENT: Des NaN subsistent après interpolation et remplissage, cela peut affecter le modèle.")
            y_series = y_series.dropna()
            print(f"  > Supprimé des lignes supplémentaires avec NaN persistants après interpolation.")

    if len(y_series) < n_splits * 2 + s_period: # Assurer une longueur minimale raisonnable pour K-Fold et auto_arima
        print(f"  ERREUR: Série '{label}' trop courte ({len(y_series)} points) pour K-Fold ou auto_arima avec s={s_period}. Impossible de modéliser.")
        # Retourne des valeurs nulles, y_series non modifiée, et ordres par défaut
        return None, None, None, y_series, None, (1,1,1), (1,1,1,s_period)

    # Initialisation des ordres par défaut si auto_arima échoue
    optimal_order = (1, 1, 1)
    optimal_seasonal_order = (1, 1, 1, s_period)

    print(f"  > Recherche des meilleurs ordres SARIMA pour '{label}' (avec auto_arima, s={s_period})...")
    try:
        arima_model = pm.auto_arima(y_series, seasonal=True, m=s_period,
                                    d=None, D=None, # auto_arima va déterminer d et D
                                    max_p=2, max_q=2, max_P=1, max_Q=1, # Plage de paramètres à tester
                                    start_p=0, start_q=0, start_P=0, start_Q=0,
                                    information_criterion='aic', # Critère de sélection (AIC, BIC, HQIC)
                                    error_action='ignore', # Ignorer les erreurs d'ajustement individuelles
                                    trace=False, suppress_warnings=True, # Pas de sortie verbeuse
                                    stepwise=True) # Utilise la méthode stepwise pour une recherche plus rapide
        
        optimal_order = arima_model.order
        optimal_seasonal_order = arima_model.seasonal_order
        print(f"  > Ordres SARIMA optimaux trouvés pour '{label}' : non-saisonnier {optimal_order}, saisonnier {optimal_seasonal_order}")
    except Exception as e:
        print(f"  ERREUR: auto_arima n'a pas pu trouver d'ordres optimaux pour '{label}': {e}. Utilisation des ordres par défaut (1,1,1)(1,1,1,{s_period}).")
        # Les valeurs par défaut sont déjà définies au début de la fonction.

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list = []
    mae_list = []

    print(f"  > Lancement de la validation croisée K-Fold ({n_splits} splits) pour '{label}' avec les ordres {optimal_order}{optimal_seasonal_order}...")
    for i, (train_idx, test_idx) in enumerate(tscv.split(y_series)):
        train, test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        print(f"    - Split {i+1}/{n_splits}: Entraînement sur {len(train)} points, Test sur {len(test)} points.")

        if len(train) == 0 or len(test) == 0:
            print(f"      AVERTISSEMENT: Train ou Test set vide pour le split {i+1} de '{label}'. Skipping.")
            continue

        try:
            model = SARIMAX(train, order=optimal_order, seasonal_order=optimal_seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            
            # S'assurer que les indices de prédiction correspondent au test set
            pred = results.predict(start=test.index[0], end=test.index[-1])
            
            # Assurez-vous que pred et test ont la même longueur et indices alignés
            common_indices = test.index.intersection(pred.index)
            if len(common_indices) == 0:
                print(f"      AVERTISSEMENT: Pas d'indices communs entre test et prédiction pour le split {i+1} de '{label}'. Skipping.")
                continue

            test_aligned = test[common_indices]
            pred_aligned = pred[common_indices]

            rmse = np.sqrt(mean_squared_error(test_aligned, pred_aligned))
            mae = mean_absolute_error(test_aligned, pred_aligned)
            rmse_list.append(rmse)
            mae_list.append(mae)
            print(f"      RMSE Split {i+1}: {rmse:.2f}, MAE Split {i+1}: {mae:.2f}")
        
        except Exception as e:
            print(f"      ERREUR dans un split ({i+1}) de '{label}': {e}. Ce split sera ignoré.")
            # Si l'erreur est liée à un problème numérique grave, on peut ajouter un break ici
            # break 

    if not rmse_list: # Si toutes les listes sont vides après les erreurs
        print(f"  AVERTISSEMENT: Aucun calcul de métrique n'a pu être effectué pour '{label}'.")
        return None, None, None, y_series, None, optimal_order, optimal_seasonal_order

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    print(f"  > Moyennes K-Fold pour '{label}': RMSE = {avg_rmse:.2f}, MAE = {avg_mae:.2f}")

    # Prédiction future après tout l'entraînement sur toutes les données disponibles
    print(f"  > Entraînement du modèle final pour '{label}' sur toutes les données pour la prévision future (avec les ordres optimaux)...")
    final_model = None # Initialiser pour le cas d'erreur
    future_forecast = pd.Series() # Initialiser une série vide
    try:
        final_model = SARIMAX(y_series, order=optimal_order, seasonal_order=optimal_seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        future_forecast = final_model.get_forecast(steps=30).predicted_mean
        print(f"  > Prévision future de 30 jours générée pour '{label}'.")
    except Exception as e:
        print(f"  ERREUR: Impossible d'entraîner le modèle final ou de générer la prévision future pour '{label}': {e}")
        # final_model reste None et future_forecast reste vide

    # Retourne la série y_series (qui peut avoir été interpolée/traitée)
    return avg_rmse, avg_mae, future_forecast, y_series, final_model, optimal_order, optimal_seasonal_order

# ==========================================
# 3. Boucle sur toutes les villes et le global
# ==========================================
def main():
    # === Paramètres ===
    path = "data/air_quality_cleaned.csv"
    output_dir = "outputsarimax_optimized"  # Nouveau dossier pour les résultats optimisés
    os.makedirs(output_dir, exist_ok=True)
    
    # Définition de la période saisonnière (m) pour auto_arima
    # S=7 pour une saisonnalité hebdomadaire (le plus probable pour données journalières)
    # S=365 pour une saisonnalité annuelle (si plusieurs années de données sont disponibles)
    s_period_for_auto_arima = 7 

    print(f"\n--- Paramètres d'exécution ---")
    print(f"Chemin du fichier de données: {path}")
    print(f"Répertoire de sortie des graphiques/résultats: {output_dir}")
    print(f"Période saisonnière (m) utilisée pour auto_arima: {s_period_for_auto_arima}")

    # === Chargement du dataset complet ===
    df = load_data(path)
    
    # --- Préparation de la série AQI Globale ---
    print(f"\n--- Préparation de la série AQI Globale ---")
    df_global = df.groupby('Date')["AQI"].mean().reset_index()
    df_global = df_global.set_index('Date').asfreq('D')
    
    missing_global_aqi = df_global['AQI'].isnull().sum()
    if missing_global_aqi > 0:
        print(f"  > {missing_global_aqi} valeurs AQI manquantes pour la série globale. Interpolation linéaire...")
        df_global['AQI'] = df_global['AQI'].interpolate(method='linear')
        df_global['AQI'] = df_global['AQI'].fillna(method='bfill').fillna(method='ffill')
        if df_global['AQI'].isnull().sum() > 0:
            print("  AVERTISSEMENT: Des NaN subsistent dans la série globale après interpolation.")
            df_global = df_global.dropna(subset=['AQI'])
            print(f"  > Supprimé des lignes supplémentaires avec NaN persistants dans la série globale.")
    
    global_series_aqi = df_global['AQI']
    print(f"Série globale préparée, longueur: {len(global_series_aqi)} points.")


    cities = df['City'].unique()
    print(f"\n--- Villes à traiter ({len(cities)} au total) + Série Globale ---")
    entities_to_process = ['Global'] + list(cities) # Ajout de 'Global' en premier

    resultats = []
    
    # === Entraînement SARIMAX avec K-Fold pour chaque entité ===
    print(f"\n--- Étape 3: Entraînement SARIMAX avec K-Fold pour chaque entité (avec optimisation des ordres) ---")
    for i, entity_name in enumerate(entities_to_process):
        print(f"\n==================================================")
        print(f"▶️ Début du traitement pour : {entity_name} ({i+1}/{len(entities_to_process)})")
        print(f"==================================================")

        current_series_for_model = None
        df_for_plotting = None
        display_name = entity_name

        if entity_name == 'Global':
            current_series_for_model = global_series_aqi
            df_for_plotting = df_global # Utiliser df_global pour le plotting de la série globale
        else:
            # Préparation des données spécifiques à la ville
            df_city_data = df[df['City'] == entity_name].copy()
            df_city_processed = df_city_data.groupby('Date')["AQI"].mean().reset_index()
            df_city_processed = df_city_processed.set_index('Date').asfreq('D')
            
            # Gérer les NaN qui pourraient apparaître après asfreq (jours manquants)
            df_city_processed['AQI'] = df_city_processed['AQI'].interpolate(method='linear')
            df_city_processed['AQI'] = df_city_processed['AQI'].fillna(method='bfill').fillna(method='ffill')
            df_city_processed = df_city_processed.dropna(subset=['AQI']) # Assurer qu'il n'y a plus de NaN
            
            current_series_for_model = df_city_processed['AQI']
            df_for_plotting = df_city_processed # Utiliser le df traité pour le plotting de la ville


        if len(current_series_for_model) < 30: # Un nombre arbitraire minimum pour la modélisation et auto_arima
            print(f"  AVERTISSEMENT: La série '{display_name}' a seulement {len(current_series_for_model)} points de données. C'est peut-être insuffisant pour SARIMAX et auto_arima.")
            # Passer à l'entité suivante si les données sont trop courtes pour être traitées
            resultats.append({"Entity": display_name, "RMSE": np.nan, "MAE": np.nan})
            print(f"❌ Traitement ignoré pour {display_name} en raison de données insuffisantes.")
            continue
            
        rmse, mae, forecast, processed_series_for_plot, model_fitted_obj, optimal_order_found, optimal_seasonal_order_found = \
            fit_sarimax_kfold(current_series_for_model, s_period=s_period_for_auto_arima, label=display_name)

        if rmse is not None and not np.isnan(rmse):
            resultats.append({"Entity": display_name, "RMSE": rmse, "MAE": mae})
            print(f"\n✅ SARIMAX K-Fold pour {display_name} terminé. RMSE final: {rmse:.2f}, MAE final: {mae:.2f}")

            # Note: processed_series_for_plot est maintenant la série y_series traitée (interpolée, etc.)
            # Si df_for_plotting est passé, c'est que c'est une DF, nous devons extraire la série AQI pour le plot
            if df_for_plotting is not None and 'AQI' in df_for_plotting.columns:
                series_to_plot_hist = df_for_plotting['AQI']
            else:
                series_to_plot_hist = processed_series_for_plot # fallback au cas où df_for_plotting n'aurait pas 'AQI'

            if not forecast.empty and not series_to_plot_hist.empty:
                print(f"  > Génération du graphique pour {display_name}...")
                plt.figure(figsize=(12, 6))
                plt.plot(series_to_plot_hist.index, series_to_plot_hist.values, label='Historique AQI', color='blue')
                
                last_hist_date = series_to_plot_hist.index[-1]
                future_index = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=len(forecast))
                
                plt.plot(future_index, forecast, label='Prévision 30j SARIMAX', linestyle='--', color='red')
                
                model_orders_str = f"SARIMA{optimal_order_found}{optimal_seasonal_order_found}"
                plt.title(f"{model_orders_str} - AQI - {display_name}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}", fontsize=14)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("AQI", fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plot_filename = f"{output_dir}/forecast_{display_name.replace(' ', '_')}_optimized.png" 
                plt.savefig(plot_filename, dpi=300)
                print(f"  > Graphique sauvegardé: {plot_filename}")
                
                print(f"  > Affichage du graphique pour {display_name}. Fermez la fenêtre pour continuer...")
                plt.show() 
                plt.close() 
            else:
                print(f"  AVERTISSEMENT: Impossible de générer le graphique pour {display_name} (prévision vide ou données historiques vides).")
        else:
            print(f"\n❌ SARIMAX K-Fold pour {display_name} a rencontré des problèmes. Les métriques ne sont pas valides.")


    # === Sauvegarde et affichage des performances finales ===
    print(f"\n--- Étape 4: Récapitulatif des performances globales ---")
    if resultats:
        df_results = pd.DataFrame(resultats)
        df_results_filename = f"{output_dir}/performances_sarimax_all_entities_optimized.csv"
        df_results.to_csv(df_results_filename, index=False)
        print(f"\n✅ Analyse terminée avec validation croisée et optimisation. Résultats agrégés sauvegardés dans : {df_results_filename}")
        print("\n--- Tableau des performances par entité (avec ordres optimisés) ---")
        print(df_results.to_string(index=False)) 
    else:
        print("\n⚠️ Aucun résultat valide à sauvegarder ou à afficher.")

    print("\nAnalyse complète terminée. Au revoir!")


if __name__ == "__main__":
    main()