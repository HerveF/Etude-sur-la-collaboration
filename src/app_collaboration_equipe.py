##==================================================================================================####
##=======   Ce code a pour objectif de créer une appli streamlit pour la visualisation  ============####
##=======   Des résultats de l'analyse des données sur la perception des élèves         ============####
## =====    vis-à-vis de la collaboration en science                                    ============####
##==================================================================================================####

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import seaborn as sns
import re
from scipy.stats import chi2_contingency
import textwrap # permet de traiter le texte
import joblib
from wordcloud import WordCloud, STOPWORDS
from unidecode import unidecode
import json, os

import sys
# permet de prendre en compte le dossier courant dans la recherche de librairies.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) 

#import src.visualizations as viz
#import src.modelisation as model

##-- On instancie les dossiers où se trouve les données
base_path = os.path.dirname("app_collaboration_equipe.py")
#folder_raw = os.path.join(root_path,"data","raw")
#folder_interim = os.path.join(root_path,"data","interim")
folder_processed = os.path.abspath(os.path.join(base_path,"..","data","processed"))
#folder_viz = os.path.join(root_path,"visualizations")


### ============================================================================================ ###
### == Étapes préalables : On récupère les différents objets dont on aura besoin par la suite == ###
### ============================================================================================ ###

# --- Configuration de la page ---
st.set_page_config(
    layout="centered",
    page_icon="📊",
    page_title="Dashbord collaboration",
    menu_items={
        'About': "Une application de démonstration pour l'analyse de données."
    }
    )
#-- On importe d'abord les fichiers contenant lesdits résultats  ---# 
@st.cache_data
def load_json(file_path):
    """Charge les données d'un ficher JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé")
        return None
    except json.JSONDecodeError:
        st.error(f"Erreur : le fichier '{file_path}' n'est pas un JSON valide")
        return None
    

# Définir un dictionnaire d'alias pour les noms de colonnes
#df_ref = load_json(os.path.join(folder_processed,'Nom des variables.csv'), index_col='Unnamed: 0').rename(
#    columns={'0': 'valeur_du_dict', 'index': 'cle_du_dict'}
#)
# Convertissez le DataFrame en dictionnaire
# Utilisez la colonne des clés comme clés et la colonne des valeurs comme valeurs
#COLUMN_ALIASES = df_ref.set_index('cle_du_dict')['valeur_du_dict'].to_dict()
file_path = os.path.abspath(os.path.join(folder_processed,'nom_des_variables.json'))
#file_path = os.path.abspath(os.path.join(base_path, '..', 'data', 'processed', 'nom_des_variables.json'))
COLUMN_ALIASES = load_json(file_path)
COLUMN_ALIASES['equipe2'] = 'Équipe corrigée'
COLUMN_ALIASES['nbPersEq2'] = 'Nombre de personnes par Équipe corrigée'
COLUMN_ALIASES['typeEq'] = "Catégorisation de l'équipe selon le genre de ses membres"
COLUMN_ALIASES['cluster'] = "Classe de l'élève selon l'étude"
COLUMN_ALIASES['genre'] = "Genre"
COLUMN_ALIASES['ecole'] = "Écoles"



# Fonction de chargement des données. On utilise st.cache_data pour la performance.
@st.cache_data
def load_data(file_name):
    """Charge un DataFrame à partir d'un fichier .parquet et retourne ses colonnes catégorielles."""
    try:
        df = pd.read_parquet(os.path.join('Data',file_name))
        return df
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé. Assurez-vous qu'il se trouve dans le même répertoire que le script.")
        return None
    
@st.cache_data
def load_all_data(parquet_files):
    """Charge tous les DataFrames à partir de fichiers .parquet dans un dictionnaire."""
    all_dataframes = {}
        
    if not parquet_files:
        st.warning("Aucun fichier .parquet n'a été trouvé dans le répertoire courant.")
        st.stop()
        
    for file_name in parquet_files:
        try:
            df = pd.read_parquet(os.path.abspath(os.path.join(folder_processed,f'{file_name}.parquet')))
            all_dataframes[file_name] = df
        except FileNotFoundError:
            st.error(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé. Assurez-vous qu'il se trouve dans le même répertoire que le script.")
            return None
    return all_dataframes


@st.cache_data
def load_joblib(file_name):
    """Charge un objet python à partir d'un fichier .joblib"""
    try:
        objet = joblib.load(os.path.abspath(os.path.join(folder_processed,file_name)))
        return objet
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{file_name}' n'a pas été trouvé. Assurez-vous qu'il se trouve dans le même répertoire que le script.")
        return None

###---- On importe directement les données nécessaires ----

# Liste des dataFrame enregistrés en parquet à importer
dfToImport = ['df_clust_CAH','df2','col_coords','col_cos2','sup_coords','sup_cos2','row_coords']
all_dataframes = load_all_data(dfToImport)

inertia_pct = load_joblib('inertia_pct.joblib')

echelle = pd.CategoricalDtype(categories=[4,2,1,3], ordered=True)
all_dataframes['df_clust_CAH']['cluster'] = all_dataframes['df_clust_CAH']['cluster'].astype(echelle)
labels_clust = {
    4: "Distraits",
    2: "Sceptiques",
    1: "Hésitants",
    3: "Convaincus"
}
all_dataframes['df_clust_CAH']['cluster'] = all_dataframes['df_clust_CAH']['cluster'].cat.rename_categories(labels_clust)
# --- Fonctions de visualisation ---

###===========================================###
#== Function pour générer le piechart stylisé ==#
###===========================================###
@st.cache_data
def pieCharts(_data: pd.DataFrame, col:str, _labels: list[str]=[], title: str="", threshold=6 , labelFont =12,
              labelInsideColors='white',figSize=(4, 4), titleFontsize=12, palette='bright'):
    """
    Creates a custum pie chart with lables inside and outside depending on the proportion limit defined

    Args:
        _data (pd.DataFrame): dataframe with col to compute
        col (str): col to compute
        title (str): Title of the piechart
        threshold (int, optionnal): percentage used to determine if the labels should be inside 
                                    (if propotion < limit) or outsite the pie. 15 by default
        lableFont (int, optionnal): font label. 12 by default
        partColors (list): colors of the piechart
        labelInsideColors (str) : Colors of the label text drawn inside the parts
        figZise : the global size of the chart. (5,5) per default
        titleFontsize (int) : default 12
        
    Returns: 
        None: Just plot the piechart
    """

    tab = _data[col].value_counts()
    sizes = tab.values
    if len(_labels) == 0:
        _labels = tab.index
    if title == "":
        title = f'Répartition des élèves suivant le {col}'
    
    partColors = sns.color_palette(palette, n_colors=len(_labels))
    

    # Calcul des pourcentages pour déterminer l'emplacement des étiquettes
    percentages = 100. * sizes / sizes.sum()

    # Création de la figure et de l'axe
    fig, ax = plt.subplots(figsize=figSize)

    # --- 3. Amélioration de la visualisation ---
    plt.title(title, fontsize=titleFontsize, fontweight='bold', pad=20, color="#181364FF")

    # Création du piechart sans étiquettes automatiques pour éviter la surcharge
    # On n'utilise ni 'labels' ni 'autopct' ici pour avoir un contrôle total
    wedges, texts = ax.pie(
        sizes,
        startangle=90,
        normalize=True,
        colors=partColors,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    # --- 4. Gestion personnalisée de l'affichage des étiquettes ---
   
    for i, p in enumerate(percentages):
        # Position des labels à l'intérieur
        if p > threshold:
            # Calcule l'angle et le rayon pour placer le texte au centre de la part
            angle = (wedges[i].theta2 + wedges[i].theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))

            # Affiche le nom du groupe et le pourcentage à l'intérieur
            ax.text(x, y, f'{(_labels[i])}\n({p:.1f}%)', color=labelInsideColors, 
                    ha='center', va='center', fontsize=labelFont, fontweight='bold')
        
        # Position des labels à l'extérieur
        else:
            # Calcule l'angle et le rayon pour placer la ligne et le texte à l'extérieur
            angle = (wedges[i].theta2 + wedges[i].theta1) / 2
            x_start = 1.0 * np.cos(np.deg2rad(angle))
            y_start = 1.0 * np.sin(np.deg2rad(angle))
            x_end = 1.20 * np.cos(np.deg2rad(angle))
            y_end = 1.20 * np.sin(np.deg2rad(angle))
            x_text = 1.25 * np.cos(np.deg2rad(angle))
            y_text = 1.25 * np.sin(np.deg2rad(angle))
            
            # Trace la ligne reliant la part au texte
            #ax.plot([wedges[i].center[0], x_start], [wedges[i].center[1], y_start], 'k-', lw=1.0)
            ax.plot([x_start, x_end], [y_start, y_end], color='k', linestyle='--', lw=0.7)
            
            # Affiche le texte à l'extérieur
            ax.text(x_text, y_text, textwrap.fill(f'{_labels[i]} ({p:.1f}%)',width=15), ha='center', va='center', fontsize=labelFont)

    # Assurez-vous que le cercle est rond
    ax.axis('equal')

    # Affichage du graphique
    plt.tight_layout()

    st.pyplot(fig)
    return None

###================================================================###
#== Function pour générer un barplot simple (une variable) stylisé ==#
###================================================================###

# Palette de couleurs en dégradé du rouge au vert
# 'RdYlGn' est une palette parfaite pour cela (Red-Yellow-Green)

# Définition du dictionnaire de labels
labels_map = {
    1: "Pas du tout d'accord",
    2: "Pas d'accord",
    3: "Je ne sais pas",
    4: "D'accord",
    5: "Tout à fait d'accord"
}

@st.cache_data
def barplot_simple(df: pd.DataFrame, col: str, titre:str):
    """
        Création de barplots simple en pourcentage pour les variables listées dans ColBarPlot.
        Args:
            df (pd.DataFrame): pd.DataFrame - DataFrame contenant les données pour le graphique
            col: str - nom de la colonne à représenter
            titre: titre à utiliser pour le graphique
            folder:str | bool - dossier dans lequel le graphique va être enregistrer au besoin. Par défaut False signifie qu'on enregistre pas le graph
        Returns: 
            None : just plot the desired graph
    """
    # Vérification de la présence de la colonne :
    if col not in df.columns:
        print(f"La colonne {col} ne figure pas dans la table df")
        return None
    
    # on utilise la palette 
    colors = sns.color_palette('RdYlGn', n_colors=df[col].nunique(dropna=True))
    # Créez une figure et un axe
    fig, ax = plt.subplots(figsize=(5,5))
    # On crée le tableau 
    tab = df[col].value_counts(normalize=True).sort_index() * 100
    # On remplace les valeurs de l'index par les labels textuels
    tab.index = tab.index.map(labels_map)   
    # on instancie la barplot
    bars = ax.bar(tab.index, tab.values, color=colors)

    # Affichez les valeurs au-dessus de chaque barre
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, # Position en X : milieu de la barre
            height + 1,                       # Position en Y : légèrement au-dessus de la barre
            f'{height:.1f}%',                 # Texte à afficher (valeur avec une décimale)
            ha='center', va='bottom',         # Alignement du texte
            fontsize=12, fontweight='bold'
        )

    # Retirez les barres du fond pour une visualisation plus épurée
    ax.yaxis.grid(False)

    # Retirez les "spines" (bordures du graphique)
    sns.despine(left=True, bottom=True)

    # Personnalisation du graphique
    #ax.set_xlabel('Modalités de réponse', fontsize=14, labelpad=15)
    ax.set_ylabel('Pourcentage (%)', fontsize=14, labelpad=15)
    #ax.set_title(f'Répartition des réponses pour la question : {col}', fontsize=18, fontweight='bold', pad=20)

    # on formate le titre avec les bons renvoi à la ligne avant de l'Ajouter au graphique.
    ax.set_title(titre, fontsize=12, fontweight='bold', pad=20,color="#181364FF")
    #ax.legend()

    # Configurez les limites de l'axe Y pour donner de l'espace au texte
    ax.set_ylim(0, max(tab.values) * 1.15)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    
    st.pyplot(fig)

    return None

###================================================================###
#== Functions pour calculer les chi2 et V_cramers de 2 col         ==#
#== Ainsi que les matrices associés pour plusieurs variables       ==#
#== Enfin créer un Heatmap associé ==#
###================================================================###

# on défini la fonction qui permet de calculer le V de cramer
@st.cache_data
def cramers_v(x, y, seuil=0.05):
    """
    Calcule le V de Cramér entre deux séries catégorielles.
    Args:
        x (pd.Series): Première série catégorielle.
        y (pd.Series): Deuxième série catégorielle.
        seuil : seuil de la p_value à considérer pour la significativité du lien
    Returns:
        float: La valeur du V de Cramér.
    """
    # Créer le tableau de contingence
    confusion_matrix = pd.crosstab(x, y)

    # Effectuer le test du Chi-2
    chi2, p_value, _, _ = chi2_contingency(confusion_matrix)

    n = confusion_matrix.sum().sum() # Nombre total d'observations
    r, k = confusion_matrix.shape    # Dimensions de la matrice (lignes, colonnes)

    # Calculer V de Cramér
    # La correction de Yates n'est pas appliquée ici par défaut par chi2_contingency
    # pour les tables > 2x2. Pour une 2x2, elle est activée si correction=True.
    # Pour le V de Cramér, nous utilisons la statistique chi2 brute.

    # Gérer les cas où min(k-1, r-1) pourrait être 0 (par exemple, si une variable n'a qu'une seule catégorie)
    if (min(k - 1, r - 1) == 0) | (p_value > seuil):
        v = 0.0 # Pas de variance, donc pas d'association significative
    else:
        phi2 = chi2 / n
        v = np.sqrt(phi2 / min(k - 1, r - 1))

    return v, p_value # Retourne le V de Cramér et la p-value


# On définit une fonction permettant de calculer les matrix de v_cramer et chi2
@st.cache_data
def v_cramer_matrix(data,colonnes):
    """
    Calcule les stats du chi2 et les v_cramer d'un ensemble de variables catéorielles
    Args:
        data (pd.dataframe) : dataframe contenant les données
        colonnes (string list): les différentes colonnes dont on doit calculer les chi2 et v de cramer
    Returns:
        cramers_v_matrix, chi2_pvalue_matrix (float matrix)  : matrices respectives des v de cramer et chi2

    """

    # on commence par vérifier que toutes les colonnes figurent bien dans le table de données
    # On ne retient que les colonnes figurant dans le tableau de données
    cols = [x for x in colonnes if x in data.columns]

    if len(cols) == 0 :
        print("Aucune des colonnes transmise ne figurent dans la table de données")
        return None
    elif len(cols) < len(colonnes) :
        removed = [x for x in colonnes if x not in cols]
        print(f"Une ou plusieurs variables entrées ne figurent pas dans la table de données. Elles ne seront pas prises en compte")
        print(f"\n Il s'agit de : {removed}")
        
    # Initialiser les matrices pour stocker les résultats
    cramers_v_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    chi2_pvalue_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    # Calculer le V de Cramér et les p-values pour toutes les paires
    print("\nCalcul des liens entre variables...")
    for col1 in cols:
         for col2 in cols:
            if col1 == col2:
                # Pour la diagonale : V de Cramér est 1 (corrélation parfaite avec soi-même)
                # Le test Chi-2 n'a pas de sens pour une variable contre elle-même
                cramers_v_matrix.loc[col1,col2] = 1.0
                chi2_pvalue_matrix.loc[col1,col2] = np.nan
            elif pd.isna(cramers_v_matrix.loc[col1,col2]): # Pour éviter les calculs en double
                v, p = cramers_v(data[col1],data[col2])

                chi2_pvalue_matrix.loc[col1,col2] = p
                chi2_pvalue_matrix.loc[col2,col1] = p

                if v==0:
                    cramers_v_matrix.loc[col1,col2] = np.nan
                    cramers_v_matrix.loc[col2,col1] = np.nan # la matrice est symétrique 
                else:    
                    cramers_v_matrix.loc[col1,col2] = v
                    cramers_v_matrix.loc[col2,col1] = v # la matrice est symétrique
   
    return cramers_v_matrix, chi2_pvalue_matrix

#fonction pour produire les heatmap avec juste les données à rentrer
@st.cache_data
def heat_v(cramers_v_matrix):
    """
        Affiche une heat matrix des v de cramers
        Args:
            cramers_v_matrix : matrix des v de cramer obtenus via la précédente fonction
            folder, file_name : à utiliser si on souhaite aussi sauvegarder le graphique
        Returns: affiche la heatmap
    """
    # --- Génération de la Heatmap pour le V de Cramér ---
    fig, ax = plt.subplots(figsize=(12, 10)) # Ajustez la taille pour une meilleure lisibilité
    size_annot_kws = round((-4/9 * cramers_v_matrix.shape[0] + 152/9),0)
    ax = sns.heatmap(
        cramers_v_matrix*100,
        annot=True,        # Afficher les valeurs dans les cellules
        fmt=".0f",         # Formater les valeurs avec 2 décimales
        cmap="magma",    # Choisir une palette de couleurs (ex: "viridis", "mako", "magma")
        linewidths=.5,     # Ajouter des lignes entre les cellules
        linecolor='black', # Couleur des lignes
        annot_kws={"size": size_annot_kws}, # Ajustez '8' à la taille de police souhaitée
        cbar_kws={'label': "V de Cramer"} # Label pour la barre de couleur
        
    )
    plt.title("Heatmap du V de Cramer (en %) entre les variables catégorielles")
    ax.xaxis.tick_top() # Place les ticks (et les étiquettes) en haut
    ax.xaxis.set_label_position('top') # S'assure que le label de l'axe X (si défini) est aussi en haut
    plt.xticks(rotation=90)

    plt.tight_layout() # Ajuste automatiquement les paramètres du graphique pour un ajustement serré
    st.pyplot(fig)

    return None

@st.cache_data
def heat_chi2(chi2_pvalue_matrix):
    """
        Affiche une heat matrix des chi2
        Args:
            chi2_pvalue_matrix : matrix des chi2 obtenus via la précédente fonction
        Return : affiche la heatmap
    """
    # --- Génération de la Heatmap pour les p-valeurs du Chi-2 ---
    fig, ax = plt.subplots(figsize=(12, 10)) # Ajustez la taille
    size_annot_kws = round((-4/9 * chi2_pvalue_matrix.shape[0] + 152/9),0)
    ax = sns.heatmap(
        chi2_pvalue_matrix,
        annot=True,        # Afficher les valeurs
        fmt=".2f",         # Formater les p-valeurs avec 3 décimales
        cmap="RdYlGn_r",   # Choisir une palette inversée (ex: "viridis_r", "YlGnBu_r").
                        # RdYlGn_r: Rouge (faible p-value, significatif) -> Jaune -> Vert (haute p-value, non significatif)
        linewidths=.5,
        linecolor='black',
        annot_kws={"size": size_annot_kws}, # Ajustez '8' à la taille de police souhaitée
        cbar_kws={'label': "P-value du Chi-2"},
        mask=chi2_pvalue_matrix.isnull() # Masquer les NaN (diagonale)
    )
    plt.title("Heatmap des P-valeurs du Test du Chi-2 entre les variables")
    ax.xaxis.tick_top() # Place les ticks (et les étiquettes) en haut
    ax.xaxis.set_label_position('top') # S'assure que le label de l'axe X (si défini) est aussi en haut
    plt.xticks(rotation=90)

    plt.tight_layout() # Ajuste automatiquement les paramètres du graphique pour un ajustement serré
    st.pyplot(fig)

    return None

###================================================================###
#== Functions pour créer les stacked bar chart         ==#
###================================================================###

##################################################################################################
#######     Fonctions permettant de créer ces barplots parlant automatiquement
##################################################################################################

# Fonction personnalisée pour vérifier la luminosité d'une couleur
def is_light(color):
    """
    Détermine si une couleur (RGB) est claire ou foncée.
    Retourne True si la couleur est claire, False sinon.
    """
    rgb_color = to_rgb(color)
    # Formule de la luminosité perçue (Luma)
    luma = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
    return luma > 0.5

### On crée le tableau croisé sur le profil ligne (réparition en ligne)
@st.cache_data
def barh(data:pd.DataFrame, col1:str, col2:str,seuil_chi2:float=0.05,palette='muted', refCol=COLUMN_ALIASES):
    """
    Trace un diagramme en barres horizontales permettant d'apprécier visuellement 
    le lien entre 2 variables qualitatives d'un Dataframe.

    Args: 
        data: Dataframe contenant les variables à représenter
        col1: Le nom de la variable à mettre en colonne
        col2: le nom de la variable qui permettra de découper les barres
        palette: le nom de la palette de couleurs seaborn à utiliser
        refCol: le dictionnaire permettant de retrouver le nom long des variables
        seuil_chi2: Seuil de significativité de la stat du chi2

    Returns: 
        None : Trace le graphique 
    """

    tab = pd.crosstab(data[col1],data[col2],normalize='index',margins=True)*100
    col1_complet = refCol[col1]
    col2_complet = refCol[col2]

    fig, ax = plt.subplots(figsize=(10, 4)) # Créer le graphique

    # Générer un diagramme en barres horizontales et empilées
    colors = sns.color_palette(palette, n_colors=tab.shape[1])
    tab.plot(kind='barh', stacked=True, ax=ax, color=colors)

    #Correction des labels
    new_labels = [textwrap.fill(label.get_text(), width=20) for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)
    
    # Ajouter le titre et les étiquettes
    # on formate le titre avec les bons renvoi à la ligne avant de l'Ajouter au graphique.
    titre = textwrap.fill(f'Comparaison des questions : {col1.capitalize()} et {col2.capitalize()}', width=45)
    ax.set_title(titre, fontsize=16, pad=20)
    ax.set_xlabel('Pourcentage (%)', fontsize=9)
    ax.set_ylabel(textwrap.fill(col1_complet, width=30), fontsize=9)

    # Positionner la légende en dehors du graphique
    ax.legend(title=f'{textwrap.fill(col2_complet,width=30)}\n', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ajouter des étiquettes de pourcentage sur chaque section de la barre
    for container in ax.containers:
        for patch in container.patches:
            width = patch.get_width()
            if width > 5: # Afficher le label uniquement si la section est suffisamment grande
                color = to_rgb(patch.get_facecolor())
                text_color = 'Black' if is_light(color) else 'white'
                ax.text(patch.get_x() + width/2, patch.get_y() + patch.get_height()/2, 
                        f'{width:.1f}%', ha='center', va='center', fontsize=9,color=text_color)

                #labels = [f'{w.get_width():.1f}%' if w.get_width() > 0.05 else '' for w in container.patches]
                #ax.bar_label(container, labels=labels, label_type='center', fontsize=9)


    # Calculer la statistique du Chi-2 et le V de Cramér
    #cramers_v_matrix, chi2_pvalue_matrix = v_cramer_matrix(data,[col1,col2])
    #chi2 = chi2_pvalue_matrix.loc[col1,col2]
    #cramer_v = cramers_v_matrix.loc[col1,col2]
    cramer_v, chi2 = cramers_v(data[col1],data[col2]) 
    
    # Interprétation des résultats
    if chi2 <= seuil_chi2:
        chi2_interpretation = "Lien confirmé à 5%" 
        xtext = 1.08
        fcolor = "#9efce0"
        if cramer_v < 0.2:
            cramer_v_interpretation = "Faible"
        elif cramer_v < 0.5:
            cramer_v_interpretation = "Modéré"
        else:
            cramer_v_interpretation = "Fort"
        # Créer le texte pour le cadre
        stats_text = (
            f"P-Value du Chi-2 : {chi2:.4f}\n"
            f"       -> {chi2_interpretation}\n\n"
            f"V de Cramer : {cramer_v*100:.1f}%\n"
            f"       -> {cramer_v_interpretation}"
        )
    else:
        chi2_interpretation = "lien non confirmé à 5%"
        xtext = 1.06
        fcolor = "#efebeb"
        # Créer le texte pour le cadre
        stats_text = (
            f"P-Value du Chi-2 : {chi2:.4f}\n"
            f"     -> {chi2_interpretation}\n"
        )

    # Ajouter le texte dans un encadré sous la légende
    ax.text(xtext, 0.15, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=fcolor, alpha=0.8, edgecolor='gray'),
            ha='left', va='center')    

    plt.tight_layout()
      
    st.pyplot(fig)

    return None


###================================================================###
#==   Fonctions de création des graphiques de l'ACM et de la CAH   ==#
###================================================================###

##---  Fonction pour tracer les individus sur un plan factoriel donné
COS2_THRESHOLD = 0.1

@st.cache_data
def plot_cluster_plan(ax1: int, ax2: int):
    """Affiche les clusters sur le plan (ax1, ax2)."""
    row_coords = all_dataframes['row_coords']
    df_cah = all_dataframes['df_clust_CAH']
    if ax1 == ax2:
        return
    if ax1 < 1 or ax2 < 1:
        return
    if ax1 > row_coords.shape[1] or ax2 > row_coords.shape[1]:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for g in sorted(df_cah['cluster'].unique()):
        mask = df_cah['cluster'] == g
        ax.scatter(
            row_coords.iloc[mask.values, ax1-1],
            row_coords.iloc[mask.values, ax2-1],
            label=f"Cluster {g}",
            alpha=0.85
        )

    ax.axhline(0, lw=1); ax.axvline(0, lw=1)
    # Étiquettes d'axes avec % d'inertie si dispo
    try:
        ax.set_xlabel(f"Axe {ax1} ({inertia_pct[ax1-1]:.2f}%)")
        ax.set_ylabel(f"Axe {ax2} ({inertia_pct[ax2-1]:.2f}%)")
    except Exception:
        ax.set_xlabel(f"Axe {ax1}")
        ax.set_ylabel(f"Axe {ax2}")

    ax.set_title(f"Plan factoriel {ax1}–{ax2} coloré par cluster (CAH)")
    ax.legend(loc='best')
    
    plt.tight_layout()
    st.pyplot(fig)


## ---- Fonction pour afficher les variables
@st.cache_data
def plot_plan2(ax1: int, ax2: int, cos2_threshold: float = COS2_THRESHOLD):

    if ax1 == ax2:
        return

    ##-- on récupère nomément les df nécessaires
    col_cos2 = all_dataframes['col_cos2']
    col_coords = all_dataframes['col_coords']
    sup_coords = all_dataframes['sup_coords']
    sup_cos2 = all_dataframes['sup_cos2']


    label1, label2 = f"Axe {ax1}", f"Axe {ax2}"

    # Garder les modalités bien représentées sur AU MOINS un des deux axes
    mask = (col_cos2[label1] >= cos2_threshold) | (col_cos2[label2] >= cos2_threshold)
    coords = col_coords.loc[mask, [label1, label2]]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[label1], coords[label2])
    for name, row in coords.iterrows():
        ax.text(row[label1], row[label2], name, fontsize=8)

    # VarSup (barycentres)
    if not sup_coords.empty:
        sup_mask = (sup_cos2[label1] >= cos2_threshold) | (sup_cos2[label2] >= cos2_threshold)
        coords_sup = sup_coords.loc[sup_mask, [label1, label2]]
        ax.scatter(coords_sup[label1], coords_sup[label2], marker='^')
        for name, row in coords_sup.iterrows():
            ax.text(row[label1], row[label2], name, fontsize=8)


    ax.axhline(0, lw=1)
    ax.axvline(0, lw=1)
    ax.set_xlabel(f"{label1} ({inertia_pct[ax1-1]:.2f}%)")
    ax.set_ylabel(f"{label2} ({inertia_pct[ax2-1]:.2f}%)")
    ax.set_title(f"Plan factoriel {ax1}-{ax2} (modalités bien représentées - prince)")

    try:
        
        handles = [Line2D([0],[0], marker='o', linestyle='None', label='Modalités actives')]
        if not sup_coords.empty:
            handles.append(Line2D([0],[0], marker='^', linestyle='None', label='VarSup (barycentres)'))
        ax.legend(handles=handles, loc='best')
    except Exception:
        pass

    plt.tight_layout()

    st.pyplot(fig)
    
###================================================================###
#== Fonctions pour la représentation des nuages de mots
###================================================================###

# Stopwords FR basiques (ajoute/retire selon ton contexte)
STOP_FR = {
    "le","la","les","un","une","des","du","de","d","au","aux","avec","sans","sur","sous",
    "et","ou","mais","donc","or","ni","car",
    "je","tu","il","elle","nous","vous","ils","elles","on","me","te","se","moi","toi","lui","leur",
    "ce","cet","cette","ces","ça","cela","ça","c","qu","que","qui","dont","où",
    "ne","pas","plus","moins","très","tres","bien","bon","bonne","aussi","ainsi","tous","tout","toutes",
    "est","sont","ai","as","avons","avez","ont","etre","être","été","etait","étais","étaient",
    "fait","fais","faisons","faites","font","peut","peux","pouvez","peuvent","pour","par","dans","en","aujourd",
    "comme","afin","chez","entre","vers","lors","lorsque","pendant",
    "oui","non","ok","daccord","d'accord","merci","svp","s'il","sil","n","a","u","x"
}
STOP_ALL = set(STOPWORDS) | STOP_FR

# Nettoyage léger : lower, enlever URLs, ponctuation, chiffres, accents → mots simples
def clean_text_series(s: pd.Series) -> list[str]:
    s = s.fillna("").astype(str).str.strip()
    s = s[s.str.len() > 0]
    # retire réponses triviales
    STOP_ANS = {"ras","r.a.s","r a s","neant","néant","aucun","/","-","n/a","na","rien","."}
    s = s[~s.str.lower().isin(STOP_ANS)]
    # nettoie chaque entrée
    out = []
    for txt in s.tolist():
        txt = re.sub(r"http[s]?://\\S+", " ", txt)     # URLs
        txt = unidecode(txt.lower())                   # accents -> ascii
        txt = re.sub(r"[^a-z\\s']", " ", txt)          # garde lettres/espace/apostrophe
        txt = re.sub(r"\\s+", " ", txt).strip()
        out.append(txt)
    return out

@st.cache_data
def make_wc(df:pd.DataFrame, col:str, title: str,
            max_words: int = 150, width: int = 1400, height: int = 900):
    texts = clean_text_series(df[col])
    if len(texts) == 0:
        print(f"[INFO] Pas de texte pour {title}")
        return None
    text_big = " ".join(texts)

    wc = WordCloud(
        width=width, height=height,
        background_color="white",
        stopwords=STOP_ALL,
        max_words=max_words,
        collocations=False  # évite d’écraser les mots uniques par des bigrams fréquents
    ).generate(text_big)

    fig, ax = plt.subplots(figsize=(width/200, height/200))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)



    

###=================================================================================###
#== Fonctions pour afficher les résultats de l'analyse des questions qualitatives   ==#
###=================================================================================###
##--- On crée la fonction qui prend en entrée un fichier JSON LLM et affiche les résultats de la manière voulue ---#

@st.cache_data
def affichage_llm(file:json, key:str, _colors=None):
    """Récupère en entrée un JSON file llm et une clé pour afficher les infos relatifs à la clé """
    contenu = file.get(key,[])
    _colors = ["#0AB19BE6","#1981BAE6","#7E7E7BE6"] if _colors==None else _colors

    #-- on affiche en premier le message principal
    main = contenu["main_message"]
    with st.container(border=True,horizontal_alignment="center"):
        #st.markdown("<h3 style='text-align: center; font-size: 25px;'> Résumé des avis recueillis </h3>",unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:{_colors[0]}; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
            f"<h4 style='font-size=25px'>Message Principal</h4>"
            f"<h5>{main}</h5>"
            f"</div>",
            unsafe_allow_html=True
        )
        #-- Ensuite on affiche les thèmes identifiés
        
        st.markdown("<h3 style='text-align: center; font-size: 25px;'> Les thèmes principaux abordés </h3>",unsafe_allow_html=True)
        themes = contenu["key_themes"]
        nThemes = len(themes)
        for line in range(0,nThemes,3):
            nbCols = min(3,(nThemes-line))
            cols = st.columns(nbCols)
            with st.container():
                for i in range(nbCols):
                    with cols[i]:
                        theme = themes[line+i]
                        st.markdown(
                                f"<div style='background-color:{_colors[1]}; padding:15px; border-radius:5px; text-align:center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
                                f"<h6>{theme}</h6>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
        
        #-- Aprés les thèmes on affiche maintenant les citations
        
        st.markdown("<h3 style='text-align: center; font-size: 25px;'> Quelques citations </h3>",unsafe_allow_html=True)
        cites = contenu["representative_quotes"]
        cols = st.columns(len(cites))

        for i, cite in enumerate(cites):
            with st.container(horizontal_alignment="center"):
                st.markdown(
                        f"<div style='background-color:{_colors[2]}; padding:5px; border-radius:5px; text-align:center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
                        f"<h6>{cite}</h6>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


###================================================================###
#==                Contenu de la page Streamlit                    ==#
###================================================================###
# ---  ---

# Titre et introduction
#st.title("Analyse de données sur l'étude de la perception des élèves sur la résolution des problèmes en équipe.")
st.title("Étude sur la collaboration")
st.markdown("<h2 style='text-align: center; color: #1E90FF; font-size: 30px;'>Étude de la perception des élèves sur la résolution des problèmes en équipe</h2>", 
            unsafe_allow_html=True)
st.markdown("""
Bienvenue sur cette application interactive d'analyse de données. 

Cette page a pour objectif de
vous permettre d'explorer les résultats de l'étude sur la perception des élèves sur la résolution 
des problèmes en équipe.
Vous pourrez visualiser les données via différents graphiques, des statistiques descriptives
aux analyses bi-variées et multivariées.

Cette étude a été réalisée par le Professeur Raoul Kamga et son équipe.
            Ces résultats sont strictement confidentiels jusqu'à publication des résultats définitifs.
""")

  
# --- Section 1 : Statistiques descriptives ---
st.header("1. Statistiques descriptives")

# Afficher le pie chart

st.subheader("Distribution en secteurs de quelques caractéristiques de l'échantillon")
# Menu déroulant pour le pie chart en utilisant les alias
colPie = ['ecole','genre','typeEq','nbPersEq2']
COLUMN_ALIASES_PIE = {key: COLUMN_ALIASES[key] for key in colPie if key in COLUMN_ALIASES.keys()}

pie_alias = st.selectbox(
    "Choisissez une variable pour avoir sa distribution",
    options=list(COLUMN_ALIASES_PIE.values()),
    key='pie_select'
)
# Trouver le nom de la colonne d'origine
pie_col = [key for key, value in COLUMN_ALIASES_PIE.items() if value == pie_alias][0]

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    if pie_col:
        if pie_col in ['ecole','genre']:
            df = all_dataframes['df_clust_CAH']
            labels = df[pie_col].unique()
            title = f'Répartition des élèves suivant la variable : {pie_col}'
            pieCharts(_data=all_dataframes['df_clust_CAH'], col=pie_col, 
                    _labels=labels, title=title, threshold=15 , labelFont =12,
                labelInsideColors='black', titleFontsize=12, palette='deep')
        else:
            tab = all_dataframes['df_clust_CAH'].groupby(pie_col, observed=False)['equipe2'].count()
            toDf =  pd.DataFrame()
            labels = tab.index
            toDf[pie_col] = [item for item, count in zip(tab.index, tab.values) for _ in range(count)]
            title = textwrap.fill(f"Répartition des équipes selon {COLUMN_ALIASES_PIE[pie_col]}", width=50)
            pieCharts(_data=toDf, col=pie_col, threshold=15 , labelFont =12, _labels=labels,title=title,
                labelInsideColors='white', titleFontsize=12, palette='deep')


# Afficher le barplot
st.subheader("Distribution des réponses aux questions d'intérêt de l'étude")
# Menu déroulant pour le barplot en utilisant les alias
colBar = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q15','q16','q17']
COLUMN_ALIASES_BAR = {key: COLUMN_ALIASES[key] for key in colBar if key in COLUMN_ALIASES.keys()}
bar_alias = st.selectbox(
    "Choisissez une variable pour le barplot",
    options=list(COLUMN_ALIASES_BAR.values()),
    key='bar_select'
)
# Trouver le nom de la colonne d'origine
bar_col = [key for key, value in COLUMN_ALIASES_BAR.items() if value == bar_alias][0]

col1, col2, col3 = st.columns([1,4,1])
with col2:
    if bar_col:
        titre = textwrap.fill(COLUMN_ALIASES_BAR[bar_col], width=50)
        barplot_simple(df=all_dataframes['df2'], col=bar_col, titre=titre)

# --- Section 2 : Analyses bi-variées ---
st.header("2. Analyses bi-variées")
st.subheader("Barplot segmenté")

# Menus déroulants pour le barplot segmenté
colBarBiv = ['ecole','typeEq','nbPersEq2'] + colBar
COLUMN_ALIASES_BAR_BIV = {key: COLUMN_ALIASES[key] for key in colBarBiv if key in COLUMN_ALIASES.keys()}
cols_bivariate_aliases = list(COLUMN_ALIASES_BAR_BIV.values())
col1 , col2 = st.columns([1,1])
with col1:
    x_alias = st.selectbox(
        "Choisissez la variable principale (axe des des ordonnées)",
        options=cols_bivariate_aliases,
        key='bivar_x'
    )
with col2:
    hue_alias = st.selectbox(
        "Choisissez la variable de segmentation (couleur)",
        options=[alias for alias in cols_bivariate_aliases if alias != x_alias],
        key='bivar_hue'
    )

# Retrouver les noms de colonnes d'origine à partir des alias
x_col = [key for key, value in COLUMN_ALIASES_BAR_BIV.items() if value == x_alias][0]
hue_col = [key for key, value in COLUMN_ALIASES_BAR_BIV.items() if value == hue_alias][0]

# Afficher le barplot segmenté
if x_col and hue_col:
    barh(data=all_dataframes['df_clust_CAH'], col1=x_col, col2=hue_col,
         seuil_chi2=0.05,palette='muted', refCol=COLUMN_ALIASES_BAR_BIV)
    
# --- Section 3 : Heatmap des corrélations ---
st.header("3. Heatmap des corrélations")

# Obtenir la liste des colonnes pour le multiselect, en utilisant les alias si disponibles
colHeatMap = ['equipe2','genre','group'] + colBarBiv
COLUMN_ALIASES_HEAT = {key: COLUMN_ALIASES[key] for key in colHeatMap if key in COLUMN_ALIASES.keys()}
col_options = [COLUMN_ALIASES_HEAT.get(col, col) for col in colHeatMap]

selected_aliases = st.multiselect(
    "Sélectionnez les variables pour la heatmap",
    options=col_options,
    key='heatmap_select'
)
# Vérification explicite pour éviter l'erreur si la liste est vide
if (not selected_aliases) | (len(selected_aliases)==1):
    st.info("Veuillez sélectionner au moins 2 variables pour la heatmap.")
else:
    # Retrouver les noms de colonnes d'origine à partir des alias sélectionnés
    selected_cols = [key for key, value in COLUMN_ALIASES_HEAT.items() if value in selected_aliases]

    # on calcule les matrices des chi2 et v de cramer
    cramers, chis = v_cramer_matrix(all_dataframes['df_clust_CAH'],selected_cols)


    if selected_cols:
        # Ici on affiche la heat map des chi2
        st.markdown("<h3 style='text-align: center; font-size: 20px;'>" \
        "Matrice des p-value des chi2 des variables sélectionnées</h3>", 
                    unsafe_allow_html=True)
        heat_chi2(chis)

        # Ensuite on affiche la heat map des V de cramer
        st.markdown("<h3 style='text-align: center; font-size: 20px;'>" \
        "Matrice des V de cramer des relations significatives à 5%</h3>", 
                    unsafe_allow_html=True)
        heat_v(cramers)

st.subheader("4. Visualisation des résultats de l'Analyse des Correspondances multiples (ACM)")
st.markdown("""Sélectionner les axes factoriels à afficher""")
_,col1,_,col2 = st.columns([1,1,1,1])
axes = range(1,11)
with col1:
    axe1 = st.selectbox(
        "Axe1",
        options=axes,
        key='axe1'
    )
with col2:
    axe2 = st.selectbox(
        "Axe2",
        options=[axe for axe in axes if axe != axe1],
        key='axe2'
    )
if (not axe1) | (not axe2):
    st.info("Veuillez sélectionner les 2 axes du plan à afficher")
else:
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>" \
        "Projection des individus sur le plan sélectionné</h3>", 
                    unsafe_allow_html=True)
    plot_cluster_plan(axe1,axe2)

    st.markdown("<h3 style='text-align: center; font-size: 20px;'>" \
        "Projection des catégories sur le plan sélectionné</h3>", 
                    unsafe_allow_html=True)
    plot_plan2(axe1,axe2,cos2_threshold=0.15)


st.header('5. Segmentation des élèves selon leur perception')
st.subheader('Réparition des élèves selon la classe')
_, col2,_ = st.columns([1,5,1])
with col2:
    pieCharts(all_dataframes['df_clust_CAH'],col='cluster',threshold=15,palette='pastel')
col1,col2 = st.columns([1,1])
with col1:
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>"
                 "Classe 1 : Les hésitants</h3>", 
                    unsafe_allow_html=True)
    st.markdown("Plutôt intéressés par la collaboration mais ils ont encore quelques réserves voire freins")
with col2:
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>"
                "Classe 2 : Les Sceptiques</h3>", 
                    unsafe_allow_html=True)
    st.markdown("Sur plusieurs aspects ils se montrent favorables tandis que sur les autres ils ne sont pas favorables")
col1,col2 = st.columns([1,1])
with col1:
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>"
                "Classe 3 : Les convaincus</h3>", 
                    unsafe_allow_html=True)
    st.markdown("Ils prêchent par l'exemple et collaborent activement avec leurs camarades.")
with col2:
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>"
                "Classe 4 : Les distraits</h3>", 
                    unsafe_allow_html=True)
    st.markdown("Dsitraits durant les cours et les présentation, ils s'alignent assez peu avec les principes de collaboration")

st.subheader('Lien entre les classes identifiées et les autres variables')
 
colCluster = ['genre','typeEq','nbPersEq2','q1','q2','q3','q4','q5','q6','q7','q8','q9',
                'q10','q11','q12','q13','q14','q15','q16','q17']
COLUMN_ALIASES_CLust = {key: COLUMN_ALIASES[key] for key in colCluster if key in COLUMN_ALIASES.keys()}
col_options = [COLUMN_ALIASES_CLust.get(col, col) for col in colCluster]
var = st.selectbox(
    'Veuiller choisir une variable à analyser vs les classes obtenues',
    options = col_options,
    key='clust'
)

if not var:
    st.info("Veullez sélectionner une variable à analyser")
else:
    # on récupère le nom de la colonne afin de créer les stacked barchart
    varSelected = [key for key, value in COLUMN_ALIASES_CLust.items() if value == var][0]
    st.markdown("<h3 style='text-align: center; font-size: 20px;'>Répartition des réponses par classe</h3>", 
                    unsafe_allow_html=True)
    palette = 'Greens' if varSelected in colBar else 'bright'
    barh(data=all_dataframes['df_clust_CAH'],col2=varSelected,col1='cluster',refCol=COLUMN_ALIASES,palette=palette)

    st.markdown("<h3 style='text-align: center; font-size: 20px;'>Répartition des cluster au sein des différentes modalités de réponse.</h3>", 
                    unsafe_allow_html=True)
    barh(data=all_dataframes['df_clust_CAH'],col1=varSelected,col2='cluster',refCol=COLUMN_ALIASES,palette='PiYG')

st.header("6. Analyse des variables qualitatives")
st.subheader("6.1 - Production des nuages de mots")
colQuali = ['quali1','quali2','quali3','quali4','quali5','quali6']
COLUMN_ALIASES_QUALI = {key: COLUMN_ALIASES[key] for key in colQuali if key in COLUMN_ALIASES.keys()}
col_options = [COLUMN_ALIASES_QUALI.get(col, col) for col in colQuali]
wcCol = st.selectbox("Choisissez la question dont vous voulez voir le nuage de mots",
             options=col_options,
             key='wc_1'
    )
wcColSelected = [key for key, value in COLUMN_ALIASES_QUALI.items() if value==wcCol][0]

make_wc(df=all_dataframes['df_clust_CAH'], col=wcColSelected, title=textwrap.fill(f"Wordcloud global – {wcCol}",width=80))
    
st.markdown("<h3 style='text-align: center; font-size: 20px;'>"
            "Wordcloud par classe identifiée</h3>", 
                    unsafe_allow_html=True)



col1, col2 = st.columns([1,1])
with col1:
    cluster = 'Convaincus'
    st.markdown(f"<h4 style='text-align: center; font-size: 16px;'>Nuage de mots du cluster des {cluster}</h4>",
                unsafe_allow_html=True)
    indices = all_dataframes['df_clust_CAH']['cluster'] == cluster
    df = all_dataframes['df_clust_CAH'][indices]
    make_wc(df=df, col=wcColSelected, title=textwrap.fill(f"Wordcloud du cluster {cluster}  – \n{wcCol}",width=80))
with col2:
    cluster = 'Hésitants'
    st.markdown(f"<h4 style='text-align: center; font-size: 16px;'>Nuage de mots du cluster des {cluster}</h4>",
                unsafe_allow_html=True)
    indices = all_dataframes['df_clust_CAH']['cluster'] == cluster
    df = all_dataframes['df_clust_CAH'][indices]
    make_wc(df=df, col=wcColSelected, title=textwrap.fill(f"Wordcloud du cluster {cluster}  – \n{wcCol}",width=80))
col1, col2 = st.columns([1,1])
with col1:
    cluster = 'Sceptiques'
    st.markdown(f"<h4 style='text-align: center; font-size: 16px;'>Nuage de mots du cluster des {cluster}</h4>",
                unsafe_allow_html=True)
    indices = all_dataframes['df_clust_CAH']['cluster'] == cluster
    df = all_dataframes['df_clust_CAH'][indices]
    make_wc(df=df, col=wcColSelected, title=textwrap.fill(f"Wordcloud du cluster {cluster}  – \n{wcCol}",width=80))
with col2:
    cluster = 'Distraits'
    st.markdown(f"<h4 style='text-align: center; font-size: 16px;'>Nuage de mots du cluster des {cluster}</h4>",
                unsafe_allow_html=True)
    indices = all_dataframes['df_clust_CAH']['cluster'] == cluster
    df = all_dataframes['df_clust_CAH'][indices]
    make_wc(df=df, col=wcColSelected, title=textwrap.fill(f"Wordcloud du cluster {cluster}  – \n{wcCol}",width=80))

#------ Affichage des résultats de l'analyse des questions ouvertes via LLM  ------#
st.subheader("6.2 - Résumés des avis exprimés par question ")


llm_general = load_json(os.path.abspath(os.path.join(folder_processed,'llm_general.json')))
llm_per_cluster = load_json(os.path.abspath(os.path.join(folder_processed,'llm_per_cluster.json')))

#-- on crée le menu déroulant des questions
llmCol = st.selectbox("Choisissez la question dont vous voulez voir le résumé des avis",
             options=col_options,
             key='llm_1'
    )
llmColSelected = [key for key, value in COLUMN_ALIASES_QUALI.items() if value==llmCol][0]
st.container()
st.markdown(f"<h4 style='text-align: center; font-size: 35px;'>Analyse globales de tous les avis exprimés sur la question</h4>",
                unsafe_allow_html=True)

affichage_llm(file=llm_general,key=llmColSelected)
#"#DD1181"
for  i, val in labels_clust.items():
    st.markdown(
        f"<div>"
        f"<h4 style='text-align: center; font-size: 25px;'>Analyse des avis exprimés sur la question pour la catégorie :</h4>"
        f"<h5 style='font-size: 30px ; text-align: center; color:#DD1181'>'{val}'</h5>"
        f"</div>",
        unsafe_allow_html=True)
    cle = f"{llmColSelected} - Cluster {i}"
    affichage_llm(file=llm_per_cluster, key=cle)
