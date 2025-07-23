import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

# --- Fonctions de Nettoyage et Chargement ---

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame individuel en :
    1. S'assurant que la colonne 'Date' est au bon format et gérant les erreurs.
    2. Extrayant des caractéristiques temporelles (année, mois, jour de la semaine).
    3. Imputant les valeurs manquantes pour les colonnes de polluants avec la médiane.

    Args:
        df (pd.DataFrame): Le DataFrame à nettoyer.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    # Important : Crée une copie pour éviter les SettingWithCopyWarning et s'assurer qu'on travaille sur une nouvelle version
    df_clean = df.copy()

    # Convertir la colonne 'Date' en datetime. Les erreurs de conversion sont transformées en NaT.
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    # Supprimer les lignes où la conversion de la date a échoué (NaT)
    df_clean.dropna(subset=['Date'], inplace=True)

    # Extraire des caractéristiques temporelles utiles pour l'analyse
    df_clean['annee'] = df_clean['Date'].dt.year
    df_clean['mois'] = df_clean['Date'].dt.month
    df_clean['jour_semaine'] = df_clean['Date'].dt.dayofweek # Lundi=0, Dimanche=6
    
    # Liste des colonnes de polluants à imputer. Adaptez cette liste si vos données contiennent d'autres colonnes.
    polluants_a_imputer = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI', 'CO2'] 
    
    # Imputation des valeurs manquantes pour chaque colonne de polluant
    for col in polluants_a_imputer:
        if col in df_clean.columns: # Vérifier si la colonne existe dans le DataFrame
            if df_clean[col].isnull().any(): # Vérifier s'il y a des valeurs manquantes dans la colonne
                median_val = df_clean[col].median() # Calculer la médiane de la colonne
                df_clean[col] = df_clean[col].fillna(median_val) # Remplacer les NaN par la médiane
    
    return df_clean

def charger_air_quality_data_seul(chemin_dossier_principal: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Charge uniquement le fichier 'air_quality.csv' depuis un dossier spécifié.
    Retourne le DataFrame brut et le DataFrame nettoyé.

    Args:
        chemin_dossier_principal (str): Le chemin vers le dossier contenant 'air_quality.csv'.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
            Un tuple contenant (DataFrame_brut, DataFrame_nettoyé).
            Retourne (None, None) en cas d'erreur ou si le fichier n'est pas trouvé.
    """
    chemin_air_quality = os.path.join(chemin_dossier_principal, 'air_quality.csv')
    
    df_raw = None
    df_clean = None

    if os.path.exists(chemin_air_quality):
        try:
            print(f"Chargement du fichier : {chemin_air_quality}")
            df_raw = pd.read_csv(chemin_air_quality) 
            print("Fichier air_quality.csv brut chargé avec succès.")
            
            print("Nettoyage du DataFrame 'air_quality'...")
            df_clean = nettoyer_donnees(df_raw.copy()) # Passer une copie pour ne pas modifier le df_raw
            print("Fichier air_quality.csv nettoyé avec succès.")
            
        except Exception as e:
            print(f"Erreur lors du traitement de air_quality.csv: {str(e)}")
    else:
        print(f"Erreur : Le fichier air_quality.csv n'a pas été trouvé à l'emplacement : {chemin_air_quality}")
    
    return df_raw, df_clean

# --- NOUVELLE FONCTION DE SAUVEGARDE ---
def sauvegarder_donnees(df: pd.DataFrame, chemin_sortie: str, nom_fichier: str) -> None:
    """
    Sauvegarde un DataFrame dans un fichier CSV.

    Args:
        df (pd.DataFrame): Le DataFrame à sauvegarder.
        chemin_sortie (str): Le chemin du dossier où enregistrer le fichier.
        nom_fichier (str): Le nom du fichier CSV (par exemple, 'air_quality_cleaned.csv').
    """
    output_path = Path(chemin_sortie)
    output_path.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas
    
    full_path = output_path / nom_fichier
    df.to_csv(full_path, index=False) # index=False pour ne pas écrire l'index du DataFrame comme une colonne
    print(f"\nDonnées nettoyées sauvegardées avec succès à : {full_path}")


# --- Classes d'Analyse EDA ---

class GlobalAnalysis:
    """
    Cette classe contient des méthodes pour effectuer une analyse globale des données (valeurs manquantes, informations générales, etc.).
    """

    @staticmethod
    def print_nan_statistics(data: pd.DataFrame, stage: str) -> None:
        """
        Affiche le nombre et le pourcentage de valeurs manquantes par colonne.

        Args:
            data (pd.DataFrame): Le DataFrame à analyser.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        print(f"\n" + f" Statistiques des valeurs manquantes ({stage}) ".center(60, "="))
        missing_info = data.isnull().sum()
        missing_percentages = (data.isnull().sum() / len(data)) * 100
        missing_df = pd.DataFrame({
            'Nombre de manquants': missing_info,
            '% de manquants': missing_percentages
        })
        # Afficher seulement les colonnes qui contiennent des valeurs manquantes
        if not missing_df[missing_df['Nombre de manquants'] > 0].empty:
            print(missing_df[missing_df['Nombre de manquants'] > 0].sort_values(by='% de manquants', ascending=False))
        else:
            print("Aucune valeur manquante détectée.")
        print("=" * 60)

    @staticmethod
    def print_info(data: pd.DataFrame, stage: str) -> None:
        """
        Affiche un résumé concis du DataFrame, incluant le type de données de chaque colonne,
        le nombre de valeurs non-nulles et l'utilisation de la mémoire.

        Args:
            data (pd.DataFrame): Le DataFrame à analyser.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        print(f"\n" + f" Informations sur le DataFrame ({stage}) ".center(60, "="))
        data.info()
        print("=" * 60)

class QuantitativeAnalysis:
    """
    Cette classe contient des méthodes pour effectuer une analyse sur les données quantitatives (statistiques descriptives, corrélation, pairplot, etc.).
    """

    @staticmethod
    def print_describe(data: pd.DataFrame, stage: str) -> None:
        """
        Affiche les statistiques descriptives (moyenne, écart-type, min, max, quartiles)
        pour toutes les colonnes numériques du DataFrame.

        Args:
            data (pd.DataFrame): Le DataFrame à analyser.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        print(f"\n" + f" Statistiques descriptives des variables numériques ({stage}) ".center(60, "="))
        print(data.describe())
        print("=" * 60)

    @staticmethod
    def plot_linear_correlation(data: pd.DataFrame, plots_dir: Path, file_prefix: str, stage: str) -> None:
        """
        Génère et sauvegarde une carte de chaleur (heatmap) des coefficients de corrélation linéaire
        (Pearson) entre les variables numériques.

        Args:
            data (pd.DataFrame): Le DataFrame contenant les données.
            plots_dir (Path): Le répertoire où sauvegarder les plots.
            file_prefix (str): Préfixe pour le nom du fichier de sortie.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        # Sélectionner uniquement les colonnes numériques pour le calcul de corrélation
        numeric_data = data.select_dtypes(include=np.number)
        
        # Exclure les colonnes temporelles extraites si elles ne sont pas pertinentes pour la corrélation directe
        cols_to_exclude_from_corr = ['annee', 'mois', 'jour_semaine']
        cols_for_corr = [col for col in numeric_data.columns if col not in cols_to_exclude_from_corr]

        if len(cols_for_corr) < 2:
            print(f"Avertissement: Pas assez de colonnes numériques pertinentes pour la matrice de corrélation ({stage}) pour {file_prefix}.")
            return

        corr = numeric_data[cols_for_corr].corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool)) # Masquer la partie supérieure de la matrice pour éviter la redondance

        plt.figure(figsize=(12, 8))
        plt.title(
            label=f"Corrélation linéaire ({stage}) pour {file_prefix}",
            fontsize=13,
            fontweight="bold",
        )
        sns.heatmap(corr, annot=True, cmap="coolwarm", mask=mask, square=True, fmt='.2f')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{file_prefix}_correlation_matrix_{stage.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_pairplot(data: pd.DataFrame, plots_dir: Path, file_prefix: str, stage: str, hue: Optional[str] = None) -> None:
        """
        Génère et sauvegarde un pairplot pour visualiser les relations bivariées
        et les distributions univariées d'un sous-ensemble de colonnes numériques.

        Args:
            data (pd.DataFrame): Le DataFrame contenant les données.
            plots_dir (Path): Le répertoire où sauvegarder les plots.
            file_prefix (str): Préfixe pour le nom du fichier de sortie.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
            hue (Optional[str]): Nom d'une colonne catégorielle à utiliser pour colorer les points.
        """
        # Sélectionner un sous-ensemble pertinent de colonnes pour le pairplot pour éviter la surcharge
        relevant_cols = ['AQI', 'CO', 'NO2', 'PM2.5', 'O3', 'PM10', 'SO2', 'CO2'] 
        existing_relevant_cols = [col for col in relevant_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

        if len(existing_relevant_cols) < 2:
            print(f"Avertissement: Pas assez de colonnes numériques pertinentes pour le pairplot ({stage}) pour {file_prefix}.")
            return
            
        # Préparer le DataFrame pour le pairplot, incluant la colonne 'hue' si elle existe et est valide
        df_plot = data[existing_relevant_cols + ([hue] if hue and hue in data.columns else [])].dropna()
        
        # Prendre un échantillon si le dataset est trop grand pour le pairplot (pour des raisons de performance)
        if len(df_plot) > 5000: # Limite arbitraire
            df_plot = df_plot.sample(n=5000, random_state=42)

        print(f"Génération du pairplot ({stage}) pour {file_prefix} ({len(df_plot)} points)... Cela peut prendre un moment.")
        
        g = sns.pairplot(
            df_plot,
            plot_kws={"alpha": 0.6}, 
            diag_kws={"fill": True, "alpha": 0.6}, 
            diag_kind="kde", 
            kind="scatter", 
            hue=hue if hue and hue in df_plot.columns else None
        )
        g.fig.suptitle(f'Pair Plot ({stage}) des Polluants et AQI pour {file_prefix}', y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f'{file_prefix}_pairplot_{stage.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

class QualitativeAnalysis:
    """
    Cette classe contient des méthodes pour effectuer une analyse sur les données qualitatives.
    """

    @staticmethod
    def print_modalities_number(data: pd.DataFrame, stage: str) -> None:
        """
        Affiche le nombre de modalités (valeurs uniques) pour chaque variable qualitative.

        Args:
            data (pd.DataFrame): Le DataFrame à analyser.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        print(f"\n" + f" Nombre de modalités par variable qualitative ({stage}) ".center(60, "="))
        # Sélectionner les colonnes de type 'object' (chaînes de caractères) ou 'category'
        qual_data = data.select_dtypes(include=['object', 'category'])
        if not qual_data.empty:
            print(qual_data.nunique(axis=0))
        else:
            print("Aucune variable qualitative trouvée.")
        print("=" * 60)

    @staticmethod
    def plot_modalities_effect_on_target(
        data: pd.DataFrame, target_column: str, qualitative_column: str, plots_dir: Path, file_prefix: str, stage: str
    ) -> None:
        """
        Génère et sauvegarde un KDE plot pour visualiser l'effet des modalités d'une
        variable qualitative sur la distribution d'une variable cible quantitative.

        Args:
            data (pd.DataFrame): Le DataFrame contenant les données.
            target_column (str): Le nom de la colonne cible quantitative.
            qualitative_column (str): Le nom de la colonne qualitative.
            plots_dir (Path): Le répertoire où sauvegarder les plots.
            file_prefix (str): Préfixe pour le nom du fichier de sortie.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        # Vérifier si les colonnes nécessaires existent
        if qualitative_column not in data.columns:
            print(f"Avertissement: La colonne qualitative '{qualitative_column}' est manquante pour {file_prefix} ({stage}).")
            return
        if target_column not in data.columns or not pd.api.types.is_numeric_dtype(data[target_column]):
            print(f"Avertissement: La colonne cible '{target_column}' est manquante ou non numérique pour {file_prefix} ({stage}).")
            return

        # Générer une palette de couleurs basée sur le nombre de modalités
        palette = sns.color_palette("viridis", n_colors=data.loc[:, qualitative_column].nunique())

        plt.figure(figsize=(12, 8))

        # Créer le KDE plot
        sns.kdeplot(
            data=data,
            x=target_column,
            hue=qualitative_column, 
            palette=palette,
            fill=True,
            alpha=0.6,
        )
        
        # Ajouter des lignes verticales pour la moyenne de la cible pour chaque modalité
        for modality in data.loc[:, qualitative_column].unique():
            mean_val = data.loc[data.loc[:, qualitative_column] == modality, target_column].mean()
            try:
                color_index = list(data.loc[:, qualitative_column].unique()).index(modality)
                color = palette[color_index]
            except ValueError:
                color = 'grey' 

            plt.axvline(
                x=mean_val,
                color=color,
                linestyle="--",
                label=f"Moyenne {modality} ({mean_val:.2f})", 
            )

        plt.title(
            label=f"Effet des modalités de {qualitative_column} sur {target_column} ({stage}) pour {file_prefix}",
            fontsize=13,
            fontweight="bold",
        )
        plt.ylabel("Densité")
        plt.xlabel(target_column)
        plt.grid(True)
        plt.legend(title=qualitative_column) 
        plt.tight_layout()
        plt.savefig(plots_dir / f'{file_prefix}_{qualitative_column}_effect_on_{target_column}_{stage.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_time_series_aqi(data: pd.DataFrame, plots_dir: Path, file_prefix: str, stage: str) -> None:
        """
        Génère et sauvegarde des plots d'évolution temporelle de l'AQI par mois et par jour de la semaine.

        Args:
            data (pd.DataFrame): Le DataFrame contenant les données.
            plots_dir (Path): Le répertoire où sauvegarder les plots.
            file_prefix (str): Préfixe pour le nom du fichier de sortie.
            stage (str): La phase d'analyse (ex: "Avant nettoyage", "Après nettoyage").
        """
        if 'AQI' not in data.columns or not pd.api.types.is_numeric_dtype(data['AQI']):
            print(f"Avertissement: Colonne 'AQI' manquante ou non numérique pour l'analyse temporelle ({stage}) pour {file_prefix}.")
            return

        if 'mois' in data.columns:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='mois', y='AQI', data=data.groupby('mois')['AQI'].mean().reset_index())
            plt.title(f'AQI Moyen par Mois ({stage}): {file_prefix}', fontsize=14, fontweight='bold')
            plt.xticks(ticks=range(1, 13), labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'])
            plt.ylabel('AQI Moyen')
            plt.xlabel('Mois')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(plots_dir / f'{file_prefix}_aqi_mensuel_{stage.lower().replace(" ", "_")}.png', dpi=300)
            plt.close()

        if 'jour_semaine' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='jour_semaine', y='AQI', data=data.groupby('jour_semaine')['AQI'].mean().reset_index())
            plt.title(f'AQI Moyen par Jour de la Semaine ({stage}): {file_prefix}', fontsize=14, fontweight='bold')
            plt.xticks(ticks=range(7), labels=['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
            plt.ylabel('AQI Moyen')
            plt.xlabel('Jour de la Semaine')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(plots_dir / f'{file_prefix}_aqi_jour_semaine_{stage.lower().replace(" ", "_")}.png', dpi=300)
            plt.close()

# --- Fonction Principale d'Exécution de l'EDA ---

def run_single_file_eda_integrated(chemin_dossier_data_principal: str):
    """
    Exécute une EDA complète pour un seul fichier de données de qualité de l'air ('air_quality.csv'),
    en effectuant l'analyse avant et après le nettoyage.

    Args:
        chemin_dossier_data_principal (str): Chemin vers le dossier contenant 'air_quality.csv'.
    """
    # Créer le répertoire de sortie pour les plots si nécessaire
    plots_dir = Path('output/plots')
    plots_dir.mkdir(parents=True, exist_ok=True) 

    # --- Chemin pour les données nettoyées ---
    cleaned_data_output_dir = chemin_dossier_data_principal # Sauvegarde dans le dossier 'data'
    cleaned_data_filename = 'air_quality_cleaned.csv' # Nom du fichier nettoyé

    print("\n" + "#" * 70)
    print(" Lancement de l'Exploratory Data Analysis (EDA) ".center(70, '#'))
    print(" sur le fichier 'air_quality.csv' ".center(70, '#'))
    print("#" * 70)
    
    # 1. Chargement du fichier air_quality.csv (brut et nettoyé)
    df_raw, df_clean = charger_air_quality_data_seul(chemin_dossier_data_principal)

    # Vérifier si les DataFrames ont été chargés correctement
    if df_raw is None or df_clean is None or df_raw.empty or df_clean.empty:
        print("\nImpossible d'exécuter l'EDA car le fichier 'air_quality.csv' n'a pas pu être chargé ou est vide.")
        return

    # --- SAUVEGARDE DES DONNÉES NETTOYÉES ICI ---
    sauvegarder_donnees(df_clean, cleaned_data_output_dir, cleaned_data_filename)

    file_prefix = "Air_Quality_Global" # Nom utilisé pour préfixer les fichiers de sortie des plots
    
    # --- PHASE 1 : EDA AVANT NETTOYAGE ---
    print("\n" + "*" * 70)
    print(" DÉBUT DE L'EDA AVANT NETTOYAGE ".center(70, '*'))
    print("*" * 70)

    # 1.1 Analyse Globale (Avant nettoyage)
    print("\n" + "=" * 20 + " Analyse Globale (Avant nettoyage) " + "=" * 20)
    GlobalAnalysis.print_info(data=df_raw, stage="Avant nettoyage")
    GlobalAnalysis.print_nan_statistics(data=df_raw, stage="Avant nettoyage")

    # 1.2 Analyse Quantitative (Avant nettoyage)
    print("\n" + "=" * 20 + " Analyse Quantitative (Avant nettoyage) " + "=" * 20)
    QuantitativeAnalysis.print_describe(data=df_raw, stage="Avant nettoyage")
    # Tenter la corrélation et le pairplot. Ils pourraient échouer si les types de données sont incohérents (chaînes au lieu de nombres)
    QuantitativeAnalysis.plot_linear_correlation(data=df_raw, plots_dir=plots_dir, file_prefix=file_prefix, stage="Avant nettoyage")
    
    # Le pairplot avant nettoyage peut être très "bruité" ou échouer si les données ne sont pas numériques
    # On peut le lancer mais s'attendre à des warnings ou des plots moins lisibles
    # Pas de 'hue' ici car 'Niveau_AQI' n'est pas encore créé
    QuantitativeAnalysis.plot_pairplot(data=df_raw, plots_dir=plots_dir, file_prefix=file_prefix, stage="Avant nettoyage")

    # 1.3 Analyse Qualitative (Avant nettoyage)
    # Les colonnes de date ne sont pas encore extraites, ni 'Niveau_AQI'
    print("\n" + "=" * 20 + " Analyse Qualitative (Avant nettoyage) " + "=" * 20)
    GlobalAnalysis.print_info(data=df_raw.select_dtypes(include=['object', 'category']), stage="Avant nettoyage - Qualitatives")
    QualitativeAnalysis.print_modalities_number(data=df_raw, stage="Avant nettoyage")
    # Les plots temporels ou les plots d'effet de modalité sur cible ne sont pas pertinents ici car les colonnes 'mois', 'jour_semaine' ou 'Niveau_AQI' n'existent pas encore ou 'Date' n'est pas toujours au bon format.

    print("\n" + "*" * 70)
    print(" FIN DE L'EDA AVANT NETTOYAGE ".center(70, '*'))
    print("*" * 70)

    # --- PHASE 2 : EDA APRÈS NETTOYAGE ---
    print("\n\n" + "*" * 70)
    print(" DÉBUT DE L'EDA APRÈS NETTOYAGE ".center(70, '*'))
    print("*" * 70)

    # 2.1 Analyse Globale (Après nettoyage)
    print("\n" + "=" * 20 + " Analyse Globale (Après nettoyage) " + "=" * 20)
    GlobalAnalysis.print_info(data=df_clean, stage="Après nettoyage")
    GlobalAnalysis.print_nan_statistics(data=df_clean, stage="Après nettoyage") # Doit montrer 0 manquants pour les polluants

    # 2.2 Analyse Quantitative (Après nettoyage)
    print("\n" + "=" * 20 + " Analyse Quantitative (Après nettoyage) " + "=" * 20)
    QuantitativeAnalysis.print_describe(data=df_clean, stage="Après nettoyage")
    QuantitativeAnalysis.plot_linear_correlation(data=df_clean, plots_dir=plots_dir, file_prefix=file_prefix, stage="Après nettoyage")
    
    # Création de 'Niveau_AQI' pour le pairplot et l'analyse qualitative post-nettoyage
    if 'AQI' in df_clean.columns:
        bins = [0, 50, 100, 150, np.inf]
        labels = ['Bon', 'Moyen', 'Mauvais', 'Très Mauvais']
        df_clean['Niveau_AQI'] = pd.cut(df_clean['AQI'], bins=bins, labels=labels, right=False)
        QuantitativeAnalysis.plot_pairplot(data=df_clean, plots_dir=plots_dir, file_prefix=file_prefix, stage="Après nettoyage", hue='Niveau_AQI')
    else:
        QuantitativeAnalysis.plot_pairplot(data=df_clean, plots_dir=plots_dir, file_prefix=file_prefix, stage="Après nettoyage")

    # 2.3 Analyse Qualitative (Après nettoyage)
    print("\n" + "=" * 20 + " Analyse Qualitative (Après nettoyage) " + "=" * 20)
    
    # Assurez-vous que 'mois' et 'jour_semaine' sont traitées comme des catégories
    df_temp_qual_clean = df_clean.copy()
    if 'mois' in df_temp_qual_clean.columns:
        df_temp_qual_clean['mois'] = df_temp_qual_clean['mois'].astype('category')
    if 'jour_semaine' in df_temp_qual_clean.columns:
        df_temp_qual_clean['jour_semaine'] = df_temp_qual_clean['jour_semaine'].astype('category')
    if 'Niveau_AQI' in df_temp_qual_clean.columns:
        df_temp_qual_clean['Niveau_AQI'] = df_temp_qual_clean['Niveau_AQI'].astype('category')

    GlobalAnalysis.print_info(data=df_temp_qual_clean.select_dtypes(include=['object', 'category']), stage="Après nettoyage - Qualitatives")
    QualitativeAnalysis.print_modalities_number(data=df_temp_qual_clean, stage="Après nettoyage")

    # Analyse de l'effet de 'Niveau_AQI' sur 'AQI'
    if 'Niveau_AQI' in df_clean.columns and 'AQI' in df_clean.columns:
        print(f"\nAnalyse de l'effet de 'Niveau_AQI' sur 'AQI' pour {file_prefix}")
        QualitativeAnalysis.plot_modalities_effect_on_target(
            data=df_clean, 
            target_column="AQI", 
            qualitative_column="Niveau_AQI", 
            plots_dir=plots_dir, 
            file_prefix=file_prefix, 
            stage="Après nettoyage"
        )

    # Analyse de l'évolution temporelle de l'AQI par mois et jour de la semaine
    QualitativeAnalysis.plot_time_series_aqi(data=df_clean, plots_dir=plots_dir, file_prefix=file_prefix, stage="Après nettoyage")

    print("\n" + "*" * 70)
    print(" FIN DE L'EDA APRÈS NETTOYAGE ".center(70, '*'))
    print("*" * 70)

    print("\n" + "#" * 70)
    print(" EDA Complète terminée. Tous les plots sont sauvegardés dans le dossier 'output/plots'. ".center(70, '#'))
    print(f" Les données nettoyées sont sauvegardées dans le dossier '{cleaned_data_output_dir}'. ".center(70, '#'))
    print("#" * 70)


# --- Point d'entrée du script ---
"""
if __name__ == "__main__":
    # Spécifiez le chemin vers le dossier qui contient votre fichier 'air_quality.csv'.
    # Exemple : Si 'air_quality.csv' est directement dans un dossier nommé 'data'
    # qui se trouve au même niveau que votre script Python, le chemin serait 'data'.
    chemin_principal_de_vos_donnees = 'data' 
    
    # Exécute l'EDA complète
    run_single_file_eda_integrated(chemin_principal_de_vos_donnees)
"""