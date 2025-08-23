## =============================================================================== ##
## == Bienvenue dans l'univers des fonctions de visualisation de notre projet   == ##
## =============================================================================== ##


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, textwrap
import json
from scipy.stats import chi2_contingency
from matplotlib.colors import to_rgb
import streamlit as st

### Initialisation des 
root_path = os.path.join("..")
folder_raw = os.path.join(root_path,"data","raw")
folder_interim = os.path.join(root_path,"data","interim")
folder_processed = os.path.join(root_path,"data","processed")
folder_viz = os.path.join(root_path,"visualizations")

# ---------------------------------------------- #
#-- Function pour générer le piechart stylisé -- #
# ---------------------------------------------- #

def pieCharts(sizes: int,labels: list[str], title: str, partColors: list, threshold=15 , labelFont =12,
              labelInsideColors='white',figSize=(5, 5), titleFontsize=12, folder:str='None', file_name='None',streamlit_plot=False):
    """
    Creates a custum pie chart with lables inside and outside depending on the proportion limit defined

    Args:
        sizes (int): values of each categories
        lables (list[str]): Labels to use
        title (str): Title of the piechart
        threshold (int, optionnal): percentage used to determine if the labels should be inside 
                                    (if propotion < limit) or outsite the pie. 15 by default
        lableFont (int, optionnal): font label. 12 by default
        partColors (list): colors of the piechart
        labelInsideColors (str) : Colors of the label text drawn inside the parts
        figZise : the global size of the chart. (5,5) per default
        titleFontsize (int) : default 12
        folder (str): the folder where the graph should be saved. By default 'None' to spécify that no save is needed
        file_name (str) : The name to file of the saved graph. if None, the name is created by default
    
    Return : Nothing. Just plot the piechart

    """

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
            ax.text(x, y, f'{(labels[i])}\n({p:.1f}%)', color=labelInsideColors, ha='center', va='center', fontsize=labelFont, fontweight='bold')
        
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
            ax.text(x_text, y_text, textwrap.fill(f'{labels[i]} ({p:.1f}%)',width=15), ha='center', va='center', fontsize=labelFont)

    # Assurez-vous que le cercle est rond
    ax.axis('equal')

    #Enregistrement du graphique si nécessaire
    if folder != 'None':
        plt.savefig(os.path.join(folder, file_name))

    # Affichage du graphique
    plt.tight_layout()

    if streamlit_plot:
        st.pyplot(fig)
    else:
        plt.show()

    return None

# ---------------------------------------------- #
#-- Barchart simple -- #
# ---------------------------------------------- #

# --- Barplot ---

# Barplots des variables .
#colBarPlot = ['groupe', 'q1', 'q2', 'q3', 'q4','q5', 'q6', 'q7', 'q8', 
#                         'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17']
# Palette de couleurs en dégradé du rouge au vert

# Définition du dictionnaire de labels
labels_map = {
    1: "Pas du tout d'accord",
    2: "Pas d'accord",
    3: "Je ne sais pas",
    4: "D'accord",
    5: "Tout à fait d'accord"
}



def barplot_simple(df: pd.DataFrame, col: str, titre:str, folder:str | bool = False, streamlit_plot = False):
    """
        Création de barplots simple en pourcentage pour les variables listées dans ColBarPlot.
        Args: 
            df (pd.DataFrame): pd.DataFrame - DataFrame contenant les données pour le graphique
            col: str - nom de la colonne à représenter
            titre: titre à utiliser pour le graphique
            folder:str | bool - dossier dans lequel le graphique va être enregistrer au besoin. 
                                Par défaut False signifie qu'on enregistre pas le graph
            streamlit_plot: indicates if the function is called by a streamlit app

        Returns: 
            None: just plot the desired graph
    """
    # Vérification de la présence de la colonne :
    if col not in df.columns:
        print(f"La colonne {col} ne figure pas dans la table df")
        return
    
    # on utilise la palette 
    colors = sns.color_palette('RdYlGn', n_colors=df[col].nunique(dropna=True))
    # Créez une figure et un axe
    fig, ax = plt.subplots(figsize=(6,6))
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
    if folder != False:
        plt.savefig(os.path.join(folder,f'{col}.png')) #on sauvegarde le fichier
    
    if streamlit_plot:
        st.pyplot(fig)
    else:
        plt.show()

    return

with open(os.path.join(folder_processed, 'nom_des_variables.json'), 'r', encoding='utf-8') as f:
    refColonnes = json.load(f)

def barplot_multiple(df: pd.DataFrame, cols: list[str], names: dict=refColonnes,folder:str | bool=False):
    """
        Création de barplots simple en pourcentage pour les variables listées dans ColBarPlot.
        Args: 
            df (pd.DataFrame): DataFrame contenant les données pour le graphique
            colBarPlot (list[str]):  Liste contenant les noms des colonnes à représenter
            names (dict, optionnal): Un dictionnaire pour renommer les titres des colonnes sur les graphiques.
                                 Les clés doivent être les noms des colonnes du DataFrame et les valeurs
                                 les intitulés désirés. La valeur par défaut utilise le dictionnaire refColonnes.
        Returns: 
            None: just plot the desired graph
    """
    # 1. on vérifie si toutes les variables figures bien dans le df
    colBarPlot = [col for col in cols if col in df.columns]
    n = len(colBarPlot)
    if n==0:
        print(f"Aucune des variables listées ne figure dans le dataFrame {df}")
        return
    elif len(colBarPlot) != len(cols):
        leftover = [col for col in cols if col not in df.columns]
        print("Une ou plusieurs variables ne figurent pas dans {df}. Il s'agit de :\n")
        print(leftover)
    
    for col in colBarPlot:
        titre = textwrap.fill(names[col], width=50)
        barplot_simple(df=df,col=col, titre=titre,folder=folder)
    
    return None
       

    
# ------------------------------------------------------------- #
#-- Calcul des stats du chi2 et matrice des p-value des chi2 -- #
# ------------------------------------------------------------- #

# on défini la fonction qui permet de calculer le V de cramer
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
def v_cramer_matrix(data,colonnes):
    """
    Calcule les stats du chi2 et les v_cramer d'un ensemble de variables catéorielles
    Args:
        data (pd.dataframe) : dataframe contenant les données
        colonnes (string list): les différentes colonnes dont on doit calculer les chi2 et v de cramer
    Return:
        cramers_v_matrix, chi2_pvalue_matrix (float matrix)  : matrices respectives des v de cramer et chi2

    """

    # on commence par vérifier que toutes les colonnes figurent bien dans le table de données
    # On ne retient que les colonnes figurant dans le tableau de données
    cols = [x for x in colonnes if x in data.columns]

    if len(cols) == 0 :
        print("Aucune des colonnes transmise ne figurent dans la table de données")
        return ''
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

                

    print("\n Matrice du V de Cramer calculées avec succès ---")
    # Le V de Cramér varie de 0 (pas d'association) à 1 (association parfaite).
    # Une valeur > 0.2-0.3 est souvent considérée comme une association faible à modérée.
    # > 0.5 forte. L'interprétation exacte dépend du contexte.
    
    print("\n--- Matrice des p-valeurs du Test du Chi-2 calculée avec succès ---")
    # Une p-valeur < 0.05 (ou autre seuil de significativité) indique une association statistiquement significative.
    # Une p-valeur élevée (par exemple > 0.05) suggère qu'il n'y a pas suffisamment de preuves pour conclure une association.
    
    return cramers_v_matrix, chi2_pvalue_matrix

#fonction pour produire les heatmap avec juste les données à rentrer

def heat_v(cramers_v_matrix, folder='None', file_name='None', streamlit_plot=False):
    """
        Affiche une heat matrix des v de cramers
        Args:
            cramers_v_matrix : matrix des v de cramer obtenus via la précédente fonction
            folder, file_name : à utiliser si on souhaite aussi sauvegarder le graphique
        Return : affiche la heatmap
    """
    # --- Génération de la Heatmap pour le V de Cramér ---
    fig, ax = plt.figure(figsize=(12, 10)) # Ajustez la taille pour une meilleure lisibilité
    ax = sns.heatmap(
        cramers_v_matrix,
        annot=True,        # Afficher les valeurs dans les cellules
        fmt=".2f",         # Formater les valeurs avec 2 décimales
        cmap="magma",    # Choisir une palette de couleurs (ex: "viridis", "mako", "magma")
        linewidths=.5,     # Ajouter des lignes entre les cellules
        linecolor='black', # Couleur des lignes
        annot_kws={"size": 8}, # Ajustez '8' à la taille de police souhaitée
        cbar_kws={'label': "V de Cramér"} # Label pour la barre de couleur
        
    )
    plt.title("Heatmap du V de Cramér entre les variables catégorielles")
    ax.xaxis.tick_top() # Place les ticks (et les étiquettes) en haut
    ax.xaxis.set_label_position('top') # S'assure que le label de l'axe X (si défini) est aussi en haut
    plt.xticks(rotation=90)

    plt.tight_layout() # Ajuste automatiquement les paramètres du graphique pour un ajustement serré

    if folder != 'None':
        plt.savefig(os.path.join(folder,file_name))

    if streamlit_plot:
        st.pyplot(fig)
    else:
        plt.show()

    return None
    

def heat_chi2(chi2_pvalue_matrix, folder='None', file_name='None', streamlit_plot=False):
    """
        Affiche une heat matrix des chi2
        Args:
            chi2_pvalue_matrix : matrix des chi2 obtenus via la précédente fonction
        streamlit_plot (Boolean): indicates if the function is called by a streamlit app. False by default
        Return : affiche la heatmap
    """
    # --- Génération de la Heatmap pour les p-valeurs du Chi-2 ---
    fig, ax = plt.figure(figsize=(12, 10)) # Ajustez la taille
    ax = sns.heatmap(
        chi2_pvalue_matrix,
        annot=True,        # Afficher les valeurs
        fmt=".2f",         # Formater les p-valeurs avec 3 décimales
        cmap="RdYlGn_r",   # Choisir une palette inversée (ex: "viridis_r", "YlGnBu_r").
                        # RdYlGn_r: Rouge (faible p-value, significatif) -> Jaune -> Vert (haute p-value, non significatif)
        linewidths=.5,
        linecolor='black',
        annot_kws={"size": 8}, # Ajustez '8' à la taille de police souhaitée
        cbar_kws={'label': "P-value du Chi-2"},
        mask=chi2_pvalue_matrix.isnull() # Masquer les NaN (diagonale)
    )
    plt.title("Heatmap des P-valeurs du Test du Chi-2 entre les variables")
    ax.xaxis.tick_top() # Place les ticks (et les étiquettes) en haut
    ax.xaxis.set_label_position('top') # S'assure que le label de l'axe X (si défini) est aussi en haut
    plt.xticks(rotation=90)

    

    plt.tight_layout() # Ajuste automatiquement les paramètres du graphique pour un ajustement serré

    if folder != 'None':
        plt.savefig(os.path.join(folder,file_name))

    if streamlit_plot:
        st.pyplot(fig)
    else:
        plt.show()
    
    return None


##################################################################################################
#######     Création de la fonction permettant de créer des stacked barplots parlant automatiquement
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

def barh(data:pd.DataFrame, col1:str, col2:str,palette='muted', refCol=refColonnes, 
         folder='None', file_name='None', show_plot=False, streamlit_plot=False ):
    """
    Trace un diagramme en barres horizontales permettant d'apprécier visuellement le lien entre 2 variables qualitatives
    d'un Dataframe.

    Args: 
        data: Dataframe contenant les variables à représenter
        col1: Le nom de la variable à mettre en colonne
        col2: le nom de la variable qui permettra de découper les barres
        palette: le nom de la palette de couleurs seaborn à utiliser
        refCol: le dictionnaire permettant de retrouver le nom long des variables
        folder: The folder where the plot should be saed if necessary. By Default 'None' if it shouldn't be saved
        file_name: The name of the plot file if necessary 
        show_plot: Boolan. If True the plot is printed
        streamlit_plot (Boolean): indicates if the function is called by a streamlit app. False by default 

    Return: Trace le graphique 
    """

    tab = pd.crosstab(data[col1],data[col2],normalize='index',margins=True)*100
    col1_complet = refCol[col1]
    col2_complet = refCol[col2]

    fig, ax = plt.subplots(figsize=(10, 4)) # Créer le graphique

    # Générer un diagramme en barres horizontales et empilées
    colors = sns.color_palette(palette, n_colors=tab.shape[1])
    tab.plot(kind='barh', stacked=True, ax=ax, color=colors)
    
    # Ajouter le titre et les étiquettes
    # on formate le titre avec les bons renvoi à la ligne avant de l'Ajouter au graphique.
    titre = textwrap.fill(f'Comparaison des questions : {col1.capitalize()} et {col2.capitalize()}', width=45)
    ax.set_title(titre, fontsize=16, pad=20)
    ax.set_xlabel('Pourcentage (%)', fontsize=9)
    ax.set_ylabel(textwrap.fill(col1_complet, width=35), fontsize=9)

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
    cramers_v_matrix, chi2_pvalue_matrix = v_cramer_matrix(data,[col1,col2])
    chi2 = chi2_pvalue_matrix.loc[col1,col2]
    cramer_v = cramers_v_matrix.loc[col1,col2]
    
    # Interprétation des résultats
    if chi2 <= 0.05:
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
    
    #We save the plot if necessary
    if folder != 'None':
        if file_name=='None':
            file_name = f'{col1} vs {col2}.png'
        plt.savefig(os.path.join(folder,file_name))
    
    if streamlit_plot:
        st.pyplot(fig)    
    elif show_plot:
        plt.show() 

    print("\nLe diagramme en barres sectionnées a été généré avec succès.")

    return None


### Test Barres verticales

def barv(data:pd.DataFrame, col1:str, col2:str,palette='muted', refCol=refColonnes, 
         folder='None', file_name='None', show_plot=False, streamlit_plot=False  ):
    """
    Trace un diagramme en barres vertical permettant d'apprécier visuellement le lien entre 2 variables qualitatives
    d'un Dataframe.

    Args : 
        data: Dataframe contenant les variables à représenter
        col1: Le nom de la variable à mettre en colonne
        col2: le nom de la variable qui permettra de découper les barres
        palette: le nom de la palette de couleurs seaborn à utiliser
        refCol: le dictionnaire permettant de retrouver le nom long des variables

    Return : Trace le graphique 
    """

    tab = pd.crosstab(data[col1],data[col2],normalize='index',margins=True)*100
    col1_complet = refCol[col1]
    col2_complet = refCol[col2]

    fig, ax = plt.subplots(figsize=(8, 6)) # Créer le graphique

    # Générer un diagramme en barres horizontales et empilées
    colors = sns.color_palette(palette, n_colors=tab.shape[1])
    tab.plot(kind='bar', stacked=True, ax=ax, color=colors)
    
    # Ajouter le titre et les étiquettes
    # on formate le titre avec les bons renvoi à la ligne avant de l'Ajouter au graphique.
    titre = textwrap.fill(f'Comparaison des questions : {col1.capitalize()} et {col2.capitalize()}', width=45)
    ax.set_title(titre, fontsize=16, pad=20)
    ax.set_ylabel('Pourcentage (%)', fontsize=9)
    ax.set_xlabel(textwrap.fill(col1_complet), fontsize=9)

    plt.xticks(rotation=45, ha='right')

    # Positionner la légende en dehors du graphique
    ax.legend(title=f'{textwrap.fill(col2_complet,width=30)}\n', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ajouter des étiquettes de pourcentage sur chaque section de la barre
    for container in ax.containers:
        for patch in container.patches:
            height = patch.get_height()
            if height > 5: # Afficher le label uniquement si la section est suffisamment grande
                color = to_rgb(patch.get_facecolor())
                text_color = 'Black' if is_light(color) else 'white'
                ax.text(patch.get_x() + patch.get_width()/2, patch.get_y() + height/2, 
                        f'{height:.1f}%', ha='center', va='center', fontsize=9,color=text_color)

                #labels = [f'{w.get_width():.1f}%' if w.get_width() > 0.05 else '' for w in container.patches]
                #ax.bar_label(container, labels=labels, label_type='center', fontsize=9)

    # Calculer la statistique du Chi-2 et le V de Cramér
    cramers_v_matrix, chi2_pvalue_matrix = v_cramer_matrix(data,[col1,col2])
    chi2 = chi2_pvalue_matrix.loc[col1,col2]
    cramer_v = cramers_v_matrix.loc[col1,col2]
    
    # Interprétation des résultats
    if chi2 <= 0.05:
        chi2_interpretation = "Lien confirmé à 5%" 
        xtext = 1.08
        fcolor = "#9efce0"
        if cramer_v < 0.1:
            cramer_v_interpretation = "Faible"
        elif cramer_v < 0.3:
            cramer_v_interpretation = "Modéré"
        else:
            cramer_v_interpretation = "Fort"
        # Créer le texte pour le cadre
        stats_text = (
            f"P-Value du Chi-2 : {chi2:.4f}\n"
            f"       -> {chi2_interpretation}\n\n"
            f"V de Cramér : {cramer_v*100:.1f}\n"
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

    if folder != 'None':
        if file_name == 'None':
            file_name = f'{col1} vs {col2} - Bar vertivales.png'
        plt.savefig(os.path.join(folder,file_name))

    if streamlit_plot:
        st.pyplot(fig)    
    elif show_plot:
        plt.show() 

    print("\nLe diagramme en barres sectionnées a été généré avec succès.")

    return None


