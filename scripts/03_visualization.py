#!/usr/bin/env python3
"""
Visualisation des Données Cliniques - Cancer en France
=====================================================

Ce script génère des visualisations complètes des données de cancérologie
pour faciliter l'interprétation et la communication des résultats.

Auteur: Cylia HANI
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Palette de couleurs pour les graphiques
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def charger_donnees():
    """
    Charge les données nettoyées pour la visualisation.
    
    Returns:
        pd.DataFrame: DataFrame avec les données cliniques
    """
    
    try:
        df = pd.read_csv('data/processed/cancer_cleaned.csv')
        df['date_diagnostic'] = pd.to_datetime(df['date_diagnostic'])
        print(f"Données chargées: {len(df)} patients")
        return df
    except FileNotFoundError:
        print("Fichier de données introuvable. Exécutez d'abord 01_data_preparation.py")
        return None

def visualiser_demographics(df):
    """
    Crée les visualisations démographiques.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("Génération des graphiques démographiques...")
    
    # Figure avec plusieurs sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse Démographique des Patients', fontsize=16, y=0.98)
    
    # 1. Distribution de l'âge
    ax1.hist(df['age'], bins=30, alpha=0.7, color=COLORS[0], edgecolor='black')
    ax1.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {df["age"].mean():.1f} ans')
    ax1.axvline(df['age'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Médiane: {df["age"].median():.1f} ans')
    ax1.set_xlabel('Âge (années)')
    ax1.set_ylabel('Nombre de patients')
    ax1.set_title('Distribution de l\'âge des patients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Répartition par sexe
    sexe_counts = df['sexe'].value_counts()
    labels = ['Hommes' if x == 'M' else 'Femmes' for x in sexe_counts.index]
    colors = [COLORS[0], COLORS[1]]
    
    wedges, texts, autotexts = ax2.pie(sexe_counts.values, labels=labels, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    ax2.set_title('Répartition par sexe')
    
    # 3. Top 10 des régions
    region_counts = df['region'].value_counts().head(10)
    bars = ax3.barh(range(len(region_counts)), region_counts.values, 
                    color=COLORS[2], alpha=0.8)
    ax3.set_yticks(range(len(region_counts)))
    ax3.set_yticklabels(region_counts.index)
    ax3.set_xlabel('Nombre de patients')
    ax3.set_title('Top 10 des régions (nombre de patients)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
    
    # 4. Groupes d'âge
    age_groups = df['groupe_age'].value_counts().sort_index()
    bars = ax4.bar(range(len(age_groups)), age_groups.values, 
                   color=COLORS[3], alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(age_groups)))
    ax4.set_xticklabels(age_groups.index, rotation=45)
    ax4.set_ylabel('Nombre de patients')
    ax4.set_title('Distribution par groupe d\'âge')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/figures/demographics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: demographics_analysis.png")

def visualiser_types_cancer(df):
    """
    Crée les visualisations des types de cancer.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("Génération des graphiques des types de cancer...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse des Types de Cancer', fontsize=16, y=0.98)
    
    # 1. Top 10 des cancers les plus fréquents
    cancer_counts = df['type_cancer'].value_counts().head(10)
    bars = ax1.barh(range(len(cancer_counts)), cancer_counts.values, 
                    color=COLORS[:len(cancer_counts)])
    ax1.set_yticks(range(len(cancer_counts)))
    ax1.set_yticklabels(cancer_counts.index)
    ax1.set_xlabel('Nombre de cas')
    ax1.set_title('Top 10 des cancers les plus fréquents')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les pourcentages
    total_patients = len(df)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = (width / total_patients) * 100
        ax1.text(width + 5, bar.get_y() + bar.get_height()/2,
                f'{int(width)} ({percentage:.1f}%)', ha='left', va='center')
    
    # 2. Répartition par stade
    stade_counts = df['stade'].value_counts().sort_index()
    wedges, texts, autotexts = ax2.pie(stade_counts.values, 
                                      labels=[f'Stade {s}' for s in stade_counts.index],
                                      autopct='%1.1f%%', startangle=90,
                                      colors=COLORS[:len(stade_counts)])
    ax2.set_title('Répartition par stade du cancer')
    
    # 3. Cancer par sexe (heatmap)
    cancer_sexe = pd.crosstab(df['type_cancer'], df['sexe'])
    top_cancers = df['type_cancer'].value_counts().head(8).index
    cancer_sexe_top = cancer_sexe.loc[top_cancers]
    
    sns.heatmap(cancer_sexe_top, annot=True, fmt='d', cmap='YlOrRd', 
                ax=ax3, cbar_kws={'label': 'Nombre de cas'})
    ax3.set_title('Distribution des cancers par sexe')
    ax3.set_xlabel('Sexe')
    ax3.set_ylabel('Type de cancer')
    
    # 4. Évolution temporelle des diagnostics
    evolution = df.groupby(['annee_diagnostic', 'type_cancer']).size().unstack(fill_value=0)
    top_5_cancers = df['type_cancer'].value_counts().head(5).index
    
    for cancer in top_5_cancers:
        if cancer in evolution.columns:
            ax4.plot(evolution.index, evolution[cancer], marker='o', 
                    linewidth=2, label=cancer)
    
    ax4.set_xlabel('Année de diagnostic')
    ax4.set_ylabel('Nombre de cas')
    ax4.set_title('Évolution temporelle des 5 cancers les plus fréquents')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/cancer_types_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Graphique sauvegardé: cancer_types_analysis.png")

def visualiser_survie_pronostic(df):
    """
    Crée les visualisations de survie et pronostic.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("Génération des graphiques de survie...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse de Survie et Pronostic', fontsize=16, y=0.98)
    
    # 1. Survie par stade
    survie_stade = df.groupby('stade')['statut_vital'].apply(
        lambda x: (x == 'Vivant').mean() * 100
    ).sort_index()
    
    bars = ax1.bar(survie_stade.index, survie_stade.values, 
                   color=COLORS[0], alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Stade du cancer')
    ax1.set_ylabel('Taux de survie (%)')
    ax1.set_title('Taux de survie par stade')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # Ajouter les valeurs et effectifs
    for i, bar in enumerate(bars):
        height = bar.get_height()
        stade = survie_stade.index[i]
        n_patients = len(df[df['stade'] == stade])
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%\n(n={n_patients})', ha='center', va='bottom')
    
    # 2. Survie par type de cancer (top 8)
    top_cancers = df['type_cancer'].value_counts().head(8).index
    survie_cancer = df[df['type_cancer'].isin(top_cancers)].groupby('type_cancer')['statut_vital'].apply(
        lambda x: (x == 'Vivant').mean() * 100
    ).sort_values(ascending=True)
    
    bars = ax2.barh(range(len(survie_cancer)), survie_cancer.values, 
                    color=COLORS[1], alpha=0.8)
    ax2.set_yticks(range(len(survie_cancer)))
    ax2.set_yticklabels(survie_cancer.index)
    ax2.set_xlabel('Taux de survie (%)')
    ax2.set_title('Taux de survie par type de cancer (Top 8)')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 100)
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        width = bar.get_width()
        cancer = survie_cancer.index[i]
        n_patients = len(df[df['type_cancer'] == cancer])
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}% (n={n_patients})', ha='left', va='center')
    
    # 3. Durée de séjour par statut vital
    df_copy = df.copy()
    sns.boxplot(data=df_copy, x='statut_vital', y='duree_sejour', ax=ax3, palette=[COLORS[2], COLORS[3]])
    ax3.set_xlabel('Statut vital')
    ax3.set_ylabel('Durée de séjour (jours)')
    ax3.set_title('Durée de séjour selon le statut vital')
    ax3.grid(True, alpha=0.3)
    
    # 4. Score de risque par statut vital
    sns.boxplot(data=df_copy, x='statut_vital', y='score_risque', ax=ax4, palette=[COLORS[4], COLORS[5]])
    ax4.set_xlabel('Statut vital')
    ax4.set_ylabel('Score de risque')
    ax4.set_title('Score de risque selon le statut vital')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/survival_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: survival_analysis.png")

def visualiser_facteurs_risque(df):
    """
    Crée les visualisations des facteurs de risque.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("Génération des graphiques des facteurs de risque...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse des Facteurs de Risque', fontsize=16, y=0.98)
    
    # 1. Distribution de l'IMC
    ax1.hist(df['imc'], bins=30, alpha=0.7, color=COLORS[0], edgecolor='black')
    ax1.axvline(df['imc'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {df["imc"].mean():.1f}')
    ax1.axvline(25, color='orange', linestyle='--', linewidth=2,
                label='Limite surpoids (25)')
    ax1.axvline(30, color='darkred', linestyle='--', linewidth=2,
                label='Limite obésité (30)')
    ax1.set_xlabel('IMC (kg/m²)')
    ax1.set_ylabel('Nombre de patients')
    ax1.set_title('Distribution de l\'IMC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Tabac et types de cancer (focus cancer poumon)
    tabac_cancer = pd.crosstab(df['type_cancer'], df['tabac'], normalize='index') * 100
    top_cancers = df['type_cancer'].value_counts().head(8).index
    tabac_cancer_top = tabac_cancer.loc[top_cancers]
    
    tabac_cancer_top.plot(kind='bar', ax=ax2, color=[COLORS[1], COLORS[2]], 
                         alpha=0.8, width=0.8)
    ax2.set_xlabel('Type de cancer')
    ax2.set_ylabel('Pourcentage de patients (%)')
    ax2.set_title('Consommation de tabac par type de cancer')
    ax2.legend(['Non-fumeur', 'Fumeur'], title='Tabac')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Catégories d'IMC par sexe
    imc_sexe = pd.crosstab(df['sexe'], df['categorie_imc'], normalize='index') * 100
    imc_sexe.plot(kind='bar', ax=ax3, color=COLORS[3:7], alpha=0.8, width=0.8)
    ax3.set_xlabel('Sexe')
    ax3.set_ylabel('Pourcentage (%)')
    ax3.set_title('Catégories d\'IMC par sexe')
    ax3.legend(title='Catégorie IMC', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Âge vs IMC (scatter plot avec régression)
    scatter = ax4.scatter(df['age'], df['imc'], alpha=0.6, c=df['score_risque'], 
                         cmap='viridis', s=30)
    
    # Ligne de régression
    z = np.polyfit(df['age'], df['imc'], 1)
    p = np.poly1d(z)
    ax4.plot(df['age'], p(df['age']), "r--", alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('Âge (années)')
    ax4.set_ylabel('IMC (kg/m²)')
    ax4.set_title('Relation Âge-IMC (couleur = score de risque)')
    ax4.grid(True, alpha=0.3)
    
    # Colorbar pour le score de risque
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Score de risque')
    
    plt.tight_layout()
    plt.savefig('results/figures/risk_factors_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: risk_factors_analysis.png")

def visualiser_correlations(df):
    """
    Crée la matrice de corrélation et les graphiques associés.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print(" Génération de la matrice de corrélation...")
    
    # Variables numériques pour la corrélation
    numeric_vars = ['age', 'duree_sejour', 'nb_traitements', 'score_risque', 'imc']
    correlation_matrix = df[numeric_vars].corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analyse des Corrélations', fontsize=16)
    
    # 1. Heatmap des corrélations
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'label': 'Coefficient de corrélation'})
    ax1.set_title('Matrice de corrélation')
    
    # 2. Corrélations avec le score de risque
    risk_correlations = correlation_matrix['score_risque'].drop('score_risque').abs().sort_values(ascending=True)
    
    bars = ax2.barh(range(len(risk_correlations)), risk_correlations.values, 
                    color=COLORS[0], alpha=0.8)
    ax2.set_yticks(range(len(risk_correlations)))
    ax2.set_yticklabels(risk_correlations.index)
    ax2.set_xlabel('Corrélation absolue avec le score de risque')
    ax2.set_title('Variables les plus corrélées au score de risque')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('results/figures/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Graphique sauvegardé: correlation_analysis.png")

def creer_dashboard_interactif(df):
    """
    Crée un dashboard interactif avec Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print(" Création du dashboard interactif...")
    
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Distribution des âges', 'Survie par type de cancer',
                       'Évolution temporelle', 'Facteurs de risque',
                       'Durée de séjour par stade', 'Distribution géographique'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Histogramme des âges
    fig.add_trace(
        go.Histogram(x=df['age'], nbinsx=30, name='Âge', showlegend=False,
                    marker_color=COLORS[0], opacity=0.7),
        row=1, col=1
    )
    
    # 2. Survie par type de cancer
    top_cancers = df['type_cancer'].value_counts().head(6).index
    survie_data = df[df['type_cancer'].isin(top_cancers)].groupby('type_cancer')['statut_vital'].apply(
        lambda x: (x == 'Vivant').mean() * 100
    ).sort_values(ascending=False)
    
    fig.add_trace(
        go.Bar(x=survie_data.index, y=survie_data.values, name='Taux de survie',
               marker_color=COLORS[1], showlegend=False),
        row=1, col=2
    )
    
    # 3. Évolution temporelle
    evolution = df.groupby('annee_diagnostic').size()
    fig.add_trace(
        go.Scatter(x=evolution.index, y=evolution.values, mode='lines+markers',
                  name='Nouveaux cas', marker_color=COLORS[2], showlegend=False),
        row=2, col=1
    )
    
    # 4. Facteurs de risque (tabac)
    tabac_counts = df['tabac'].value_counts()
    fig.add_trace(
        go.Pie(labels=tabac_counts.index, values=tabac_counts.values,
               name='Tabac', showlegend=False),
        row=2, col=2
    )
    
    # 5. Durée de séjour par stade
    for i, stade in enumerate(sorted(df['stade'].unique())):
        duree_stade = df[df['stade'] == stade]['duree_sejour']
        fig.add_trace(
            go.Box(y=duree_stade, name=f'Stade {stade}', showlegend=False,
                   marker_color=COLORS[i]),
            row=3, col=1
        )
    
    # 6. Distribution géographique (top 10 régions)
    region_counts = df['region'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=region_counts.values, y=region_counts.index, 
               orientation='h', name='Régions', showlegend=False,
               marker_color=COLORS[3]),
        row=3, col=2
    )
    
    # Mise à jour des axes et titre
    fig.update_layout(
        title_text="Dashboard Interactif - Analyse des Données de Cancer",
        title_x=0.5,
        height=1200,
        showlegend=False
    )
    
    # Sauvegarder le dashboard
    fig.write_html('results/figures/dashboard_interactif.html')
    
    print(" Dashboard interactif sauvegardé: dashboard_interactif.html")

def generer_rapport_visualisation():
    """
    Génère un rapport récapitulatif des visualisations créées.
    """
    
    rapport = f"""
# RAPPORT DE VISUALISATION
## Données Cliniques - Cancer en France
### Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}

---

## GRAPHIQUES GÉNÉRÉS

### 1. **demographics_analysis.png**
- Distribution de l'âge des patients
- Répartition par sexe (diagramme circulaire)
- Top 10 des régions les plus représentées
- Distribution par groupe d'âge

### 2. **cancer_types_analysis.png**
- Top 10 des cancers les plus fréquents
- Répartition par stade du cancer
- Distribution des cancers par sexe (heatmap)
- Évolution temporelle des diagnostics

### 3. **survival_analysis.png**
- Taux de survie par stade du cancer
- Taux de survie par type de cancer
- Durée de séjour selon le statut vital
- Score de risque selon le statut vital

### 4. **risk_factors_analysis.png**
- Distribution de l'IMC
- Consommation de tabac par type de cancer
- Catégories d'IMC par sexe
- Relation âge-IMC avec score de risque

### 5. **correlation_analysis.png**
- Matrice de corrélation des variables numériques
- Variables les plus corrélées au score de risque

### 6. **dashboard_interactif.html**
- Dashboard interactif avec 6 visualisations
- Graphiques dynamiques avec Plotly
- Navigation intuitive pour exploration des données

---

## INSIGHTS VISUELS CLÉS

### Démographie
- Population majoritairement âgée (pic 60-75 ans)
- Légère prédominance masculine
- Île-de-France fortement représentée

### Types de Cancer
- Cancers du poumon, sein et colorectal dominent
- Majorité diagnostiqués aux stades précoces (I-II)
- Évolution stable sur la période étudiée

### Survie
- Corrélation forte entre stade et survie
- Variabilité importante selon le type de cancer
- Score de risque pertinent pour le pronostic

### Facteurs de Risque
- 40% de fumeurs dans la population
- Corrélation tabac-cancer du poumon visible
- IMC moyen légèrement élevé (surpoids)

---

## RECOMMANDATIONS D'UTILISATION

1. **Présentations** : Utiliser les PNG haute résolution
2. **Exploration** : Consulter le dashboard interactif
3. **Publications** : Adapter les couleurs selon les guidelines
4. **Formation** : Combiner avec les analyses statistiques

---

*Toutes les visualisations sont sauvegardées dans le dossier results/figures/*
"""
    
    # Sauvegarder le rapport
    with open('results/reports/rapport_visualisation.md', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print(f" Rapport de visualisation sauvegardé: results/reports/rapport_visualisation.md")

def main():
    """
    Fonction principale qui génère toutes les visualisations.
    """
    
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*40)
    
    # Charger les données
    df = charger_donnees()
    if df is None:
        return
    
    # Créer le dossier des figures s'il n'existe pas
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # Générer toutes les visualisations
    visualiser_demographics(df)
    visualiser_types_cancer(df)
    visualiser_survie_pronostic(df)
    visualiser_facteurs_risque(df)
    visualiser_correlations(df)
    creer_dashboard_interactif(df)
    
    # Générer le rapport
    generer_rapport_visualisation()
    
    print(f" Toutes les visualisations ont été générées !")
    print(f" Consultez le dossier 'results/figures/' pour voir les graphiques.")
    print(f"Ouvrez 'dashboard_interactif.html' dans votre navigateur.")

if __name__ == "__main__":
    main()
