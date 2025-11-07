#!/usr/bin/env python3
"""
Analyse Exploratoire des Données Cliniques - Cancer en France
============================================================

Ce script effectue une analyse exploratoire complète des données de cancérologie,
incluant des statistiques descriptives, des analyses de corrélation et
des insights cliniques.

Auteur: Cylia HANI
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Créer le dossier des résultats
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/reports', exist_ok=True)

def charger_donnees():
    """
    Charge les données nettoyées.
    
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

def analyser_demographics(df):
    """
    Analyse les caractéristiques démographiques de la population.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print("ANALYSE DÉMOGRAPHIQUE")
    print("="*60)
    
    # Analyse de l'âge
    print(f"ANALYSE DE L'ÂGE")
    print("-" * 30)
    print(f"Âge moyen: {df['age'].mean():.1f} ± {df['age'].std():.1f} ans")
    print(f"Âge médian: {df['age'].median():.1f} ans")
    print(f"Étendue: {df['age'].min()} - {df['age'].max()} ans")
    
    print(f"\nRépartition par groupe d'âge:")
    age_groups = df['groupe_age'].value_counts().sort_index()
    for groupe, count in age_groups.items():
        percentage = (count / len(df)) * 100
        print(f"  {groupe}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Analyse par sexe
    print(f"\n⚧ RÉPARTITION PAR SEXE")
    print("-" * 30)
    sexe_counts = df['sexe'].value_counts()
    for sexe, count in sexe_counts.items():
        percentage = (count / len(df)) * 100
        sexe_label = "Hommes" if sexe == 'M' else "Femmes"
        print(f"  {sexe_label}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Analyse régionale
    print(f" RÉPARTITION GÉOGRAPHIQUE")
    print("-" * 30)
    region_counts = df['region'].value_counts().head(10)
    for region, count in region_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {region:<25}: {count:4d} ({percentage:4.1f}%)")

def analyser_cancers(df):
    """
    Analyse les types de cancer et leur distribution.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print(" ANALYSE DES TYPES DE CANCER")
    print("="*60)
    
    # Types de cancer les plus fréquents
    print(f"CANCERS LES PLUS FRÉQUENTS")
    print("-" * 40)
    cancer_counts = df['type_cancer'].value_counts()
    for i, (cancer, count) in enumerate(cancer_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  {i:2d}. {cancer:<15}: {count:4d} cas ({percentage:4.1f}%)")
    
    # Analyse par stade
    print(f"RÉPARTITION PAR STADE")
    print("-" * 30)
    stade_counts = df['stade'].value_counts().sort_index()
    for stade, count in stade_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Stade {stade}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Analyse des traitements
    print(f"TRAITEMENTS LES PLUS UTILISÉS")
    print("-" * 35)
    all_treatments = []
    for treatments in df['traitement']:
        if treatments != 'Surveillance':
            all_treatments.extend([t.strip() for t in treatments.split(';')])
    
    treatment_counts = pd.Series(all_treatments).value_counts()
    for treatment, count in treatment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {treatment:<20}: {count:4d} patients ({percentage:4.1f}%)")

def analyser_survie_pronostic(df):
    """
    Analyse les facteurs de survie et de pronostic.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print("ANALYSE DE SURVIE ET PRONOSTIC")
    print("="*60)
    
    # Statut vital global
    survie_counts = df['statut_vital'].value_counts()
    taux_survie = (survie_counts['Vivant'] / len(df)) * 100
    
    print(f" STATUT VITAL GLOBAL")
    print("-" * 25)
    print(f"  Taux de survie global: {taux_survie:.1f}%")
    for statut, count in survie_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {statut}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Survie par stade
    print(f"\n📈 SURVIE PAR STADE")
    print("-" * 20)
    survie_stade = df.groupby('stade')['statut_vital'].apply(
        lambda x: (x == 'Vivant').mean() * 100
    ).sort_index()
    
    for stade, taux in survie_stade.items():
        nb_patients = len(df[df['stade'] == stade])
        print(f"  Stade {stade}: {taux:5.1f}% (n={nb_patients})")
    
    # Survie par type de cancer
    print(f"SURVIE PAR TYPE DE CANCER (Top 5)")
    print("-" * 35)
    top_cancers = df['type_cancer'].value_counts().head(5).index
    survie_cancer = df[df['type_cancer'].isin(top_cancers)].groupby('type_cancer')['statut_vital'].apply(
        lambda x: (x == 'Vivant').mean() * 100
    ).sort_values(ascending=False)
    
    for cancer, taux in survie_cancer.items():
        nb_patients = len(df[df['type_cancer'] == cancer])
        print(f"  {cancer:<15}: {taux:5.1f}% (n={nb_patients})")

def analyser_facteurs_risque(df):
    """
    Analyse les facteurs de risque et comorbidités.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print("ANALYSE DES FACTEURS DE RISQUE")
    print("="*60)
    
    # Tabac
    print(f" CONSOMMATION DE TABAC")
    print("-" * 25)
    tabac_counts = df['tabac'].value_counts()
    for tabac, count in tabac_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {tabac}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Alcool
    print(f"CONSOMMATION D'ALCOOL")
    print("-" * 25)
    alcool_counts = df['alcool'].value_counts()
    for alcool, count in alcool_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {alcool:<12}: {count:4d} patients ({percentage:4.1f}%)")
    
    # IMC
    print(f"INDICE DE MASSE CORPORELLE")
    print("-" * 30)
    print(f"IMC moyen: {df['imc'].mean():.1f} ± {df['imc'].std():.1f}")
    imc_categories = df['categorie_imc'].value_counts()
    for category, count in imc_categories.items():
        percentage = (count / len(df)) * 100
        print(f"  {category:<12}: {count:4d} patients ({percentage:4.1f}%)")
    
    # Comorbidités
    print(f" COMORBIDITÉS")
    print("-" * 15)
    patients_avec_comorbidites = df[df['comorbidites'] != 'Aucune']
    pourcentage_comorbidites = (len(patients_avec_comorbidites) / len(df)) * 100
    
    print(f"Patients avec comorbidités: {len(patients_avec_comorbidites)} ({pourcentage_comorbidites:.1f}%)")
    
    # Extraire les comorbidités individuelles
    all_comorbidites = []
    for comorbidite_list in df[df['comorbidites'] != 'Aucune']['comorbidites']:
        all_comorbidites.extend([c.strip() for c in comorbidite_list.split(';')])
    
    if all_comorbidites:
        comorbidite_counts = pd.Series(all_comorbidites).value_counts()
        for comorbidite, count in comorbidite_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {comorbidite:<25}: {count:4d} patients ({percentage:4.1f}%)")

def analyser_tendances_temporelles(df):
    """
    Analyse les tendances temporelles des diagnostics.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print(" ANALYSE TEMPORELLE")
    print("="*60)
    
    # Diagnostics par année
    print(f" ÉVOLUTION ANNUELLE DES DIAGNOSTICS")
    print("-" * 40)
    diagnostics_annee = df['annee_diagnostic'].value_counts().sort_index()
    for annee, count in diagnostics_annee.items():
        print(f"  {annee}: {count:4d} diagnostics")
    
    # Saisonnalité (par mois)
    print(f"SAISONNALITÉ DES DIAGNOSTICS")
    print("-" * 30)
    mois_noms = {1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
                 5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
                 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'}
    
    diagnostics_mois = df['mois_diagnostic'].value_counts().sort_index()
    for mois, count in diagnostics_mois.items():
        print(f"  {mois_noms[mois]:<12}: {count:4d} diagnostics")

def analyser_correlations(df):
    """
    Analyse les corrélations entre les variables numériques.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print("ANALYSE DES CORRÉLATIONS")
    print("="*60)
    
    # Variables numériques pour l'analyse de corrélation
    numeric_vars = ['age', 'duree_sejour', 'nb_traitements', 'score_risque', 'imc']
    
    # Calculer la matrice de corrélation
    correlation_matrix = df[numeric_vars].corr()
    
    print(f" CORRÉLATIONS SIGNIFICATIVES (|r| > 0.3)")
    print("-" * 45)
    
    # Trouver les corrélations significatives
    for i, var1 in enumerate(numeric_vars):
        for j, var2 in enumerate(numeric_vars):
            if i < j:  # Éviter les doublons
                corr = correlation_matrix.loc[var1, var2]
                if abs(corr) > 0.3:
                    direction = "positive" if corr > 0 else "négative"
                    print(f"  {var1} ↔ {var2}: r = {corr:.3f} ({direction})")
    
    return correlation_matrix

def tests_statistiques(df):
    """
    Effectue des tests statistiques pour identifier des associations significatives.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    """
    
    print("\n" + "="*60)
    print("TESTS STATISTIQUES")
    print("="*60)
    
    # Test 1: Âge moyen selon le statut vital
    print(f"TEST 1: Âge et statut vital")
    print("-" * 30)
    
    age_vivant = df[df['statut_vital'] == 'Vivant']['age']
    age_decede = df[df['statut_vital'] == 'Décédé']['age']
    
    stat, p_value = stats.ttest_ind(age_vivant, age_decede)
    
    print(f"Âge moyen (vivants): {age_vivant.mean():.1f} ans")
    print(f"Âge moyen (décédés): {age_decede.mean():.1f} ans")
    print(f"Test t de Student: t = {stat:.3f}, p = {p_value:.3f}")
    
    if p_value < 0.05:
        print(" Différence significative (p < 0.05)")
    else:
        print("Pas de différence significative (p ≥ 0.05)")
    
    # Test 2: Association tabac et cancer du poumon
    print(f"TEST 2: Tabac et cancer du poumon")
    print("-" * 35)
    
    # Créer un tableau de contingence
    contingency_table = pd.crosstab(
        df['type_cancer'] == 'Poumon', 
        df['tabac'] == 'Oui'
    )
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"Test du Chi² d'indépendance:")
    print(f"Chi² = {chi2:.3f}, p = {p_value:.3f}")
    
    if p_value < 0.05:
        print("Association significative (p < 0.05)")
    else:
        print("Pas d'association significative (p ≥ 0.05)")

def generer_rapport_analyse(df, correlation_matrix):
    """
    Génère un rapport d'analyse détaillé.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
        correlation_matrix (pd.DataFrame): Matrice de corrélation
    """
    
    rapport = f"""
# RAPPORT D'ANALYSE EXPLORATOIRE
## Données Cliniques - Cancer en France
### Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}

---

## RÉSUMÉ EXÉCUTIF

Cette analyse porte sur {len(df)} patients diagnostiqués avec un cancer entre {df['date_diagnostic'].min().strftime('%Y')} et {df['date_diagnostic'].max().strftime('%Y')}.

### Points Clés:
- **Âge moyen**: {df['age'].mean():.1f} ans
- **Taux de survie global**: {((df['statut_vital'] == 'Vivant').mean() * 100):.1f}%
- **Cancer le plus fréquent**: {df['type_cancer'].value_counts().index[0]}
- **Durée moyenne de séjour**: {df['duree_sejour'].mean():.1f} jours

---

## DONNÉES DÉMOGRAPHIQUES

### Répartition par sexe:
{df['sexe'].value_counts().to_string()}

### Répartition par groupe d'âge:
{df['groupe_age'].value_counts().sort_index().to_string()}

---

## TYPES DE CANCER

### Top 10 des cancers les plus fréquents:
{df['type_cancer'].value_counts().head(10).to_string()}

### Répartition par stade:
{df['stade'].value_counts().sort_index().to_string()}

---

## FACTEURS DE RISQUE

### Tabac:
{df['tabac'].value_counts().to_string()}

### Alcool:
{df['alcool'].value_counts().to_string()}

### IMC:
{df['categorie_imc'].value_counts().to_string()}

---

## PRONOSTIC

### Survie par stade:
{df.groupby('stade')['statut_vital'].apply(lambda x: f"{(x == 'Vivant').mean() * 100:.1f}%").to_string()}

### Survie par type de cancer (Top 5):
{df[df['type_cancer'].isin(df['type_cancer'].value_counts().head(5).index)].groupby('type_cancer')['statut_vital'].apply(lambda x: f"{(x == 'Vivant').mean() * 100:.1f}%").sort_values(ascending=False).to_string()}

---

## CORRÉLATIONS

Matrice de corrélation des variables numériques:
{correlation_matrix.round(3).to_string()}

---

## RECOMMANDATIONS

1. **Dépistage précoce**: {((df['stade'].isin(['I', 'II'])).mean() * 100):.1f}% des cancers sont diagnostiqués aux stades I-II
2. **Facteurs de risque**: {((df['tabac'] == 'Oui').mean() * 100):.1f}% des patients sont fumeurs
3. **Prise en charge**: Durée moyenne de séjour de {df['duree_sejour'].mean():.1f} jours

---

*Rapport généré automatiquement par le script d'analyse exploratoire*
"""
    
    # Sauvegarder le rapport
    with open('results/reports/rapport_analyse_exploratoire.md', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print(f"\n📄 Rapport détaillé sauvegardé: results/reports/rapport_analyse_exploratoire.md")

def main():
    """
    Fonction principale qui orchestre l'analyse exploratoire.
    """
    
    print(" ANALYSE EXPLORATOIRE DES DONNÉES CLINIQUES")
    print("="*60)
    
    # Charger les données
    df = charger_donnees()
    if df is None:
        return
    
    # Analyses
    analyser_demographics(df)
    analyser_cancers(df)
    analyser_survie_pronostic(df)
    analyser_facteurs_risque(df)
    analyser_tendances_temporelles(df)
    correlation_matrix = analyser_correlations(df)
    tests_statistiques(df)
    
    # Générer le rapport
    generer_rapport_analyse(df, correlation_matrix)
    
    print(f"Analyse exploratoire terminée !")
    print(f" Consultez le dossier 'results/' pour les résultats détaillés.")

if __name__ == "__main__":
    main()
