#!/usr/bin/env python3
"""
Script de Préparation des Données Cliniques - Cancer en France
=============================================================

Ce script génère et prépare les données cliniques simulées basées sur
les statistiques officielles de l'Institut National du Cancer (INCa).

Données basées sur :
- 433 000 nouveaux cas de cancer en France en 2023
- Répartition par type de cancer selon les statistiques françaises
- Facteurs démographiques réalistes

Auteur: Cylia HANI
Date: 2024
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from faker import Faker
import random

# Configuration
fake = Faker('fr_FR')  # Localization française
np.random.seed(42)  # Reproductibilité
random.seed(42)

# Créer les dossiers nécessaires
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def generer_donnees_cancer():
    """
    Génère des données cliniques simulées basées sur les statistiques françaises.
    
    Returns:
        pd.DataFrame: DataFrame avec les données cliniques
    """
    
    # Paramètres basés sur les statistiques françaises 2023
    nb_patients = 5000  # Échantillon représentatif
    
    # Types de cancer et leur fréquence (basé sur les données INCa)
    types_cancer = {
        'Poumon': 0.20,      # Le plus fréquent
        'Sein': 0.18,
        'Colorectal': 0.15,
        'Prostate': 0.12,
        'Vessie': 0.08,
        'Rein': 0.06,
        'Foie': 0.05,
        'Estomac': 0.04,
        'Pancréas': 0.04,
        'Autres': 0.08
    }
    
    # Régions françaises avec population approximative
    regions = {
        'Île-de-France': 0.18,
        'Auvergne-Rhône-Alpes': 0.12,
        'Hauts-de-France': 0.09,
        'Nouvelle-Aquitaine': 0.09,
        'Occitanie': 0.09,
        'Grand Est': 0.08,
        'Provence-Alpes-Côte d\'Azur': 0.08,
        'Pays de la Loire': 0.06,
        'Normandie': 0.05,
        'Bretagne': 0.05,
        'Centre-Val de Loire': 0.04,
        'Bourgogne-Franche-Comté': 0.04,
        'Corse': 0.01,
        'DOM-TOM': 0.02
    }
    
    # Génération des données
    donnees = []
    
    print(f"Génération de {nb_patients} dossiers patients...")
    
    for i in range(nb_patients):
        # Informations démographiques
        sexe = np.random.choice(['M', 'F'], p=[0.54, 0.46])  # Légèrement plus d'hommes
        
        # Distribution d'âge réaliste pour le cancer (pic à 65-75 ans)
        age = int(np.random.gamma(7, 9) + 30)
        age = min(max(age, 25), 95)  # Limiter entre 25 et 95 ans
        
        # Type de cancer influencé par le sexe
        if sexe == 'F':
            # Plus de cancers du sein, moins de prostate chez les femmes
            types_ajustes = {
                'Sein': 0.35,
                'Poumon': 0.15,
                'Colorectal': 0.15,
                'Ovaire': 0.08,
                'Utérus': 0.07,
                'Vessie': 0.04,
                'Rein': 0.05,
                'Autres': 0.11
            }
        else:
            types_ajustes = {
                'Prostate': 0.25,
                'Poumon': 0.25,
                'Colorectal': 0.15,
                'Vessie': 0.12,
                'Rein': 0.07,
                'Foie': 0.06,
                'Estomac': 0.05,
                'Autres': 0.05
            }
        
        type_cancer = np.random.choice(
            list(types_ajustes.keys()), 
            p=list(types_ajustes.values())
        )
        
        # Région
        region = np.random.choice(
            list(regions.keys()), 
            p=list(regions.values())
        )
        
        # Date de diagnostic (dernières 5 années)
        date_debut = datetime.now() - timedelta(days=5*365)
        date_diagnostic = fake.date_between(
            start_date=date_debut, 
            end_date=datetime.now()
        )
        
        # Stade du cancer (I-IV)
        # Répartition réaliste : plus de stades précoces grâce au dépistage
        stade = np.random.choice(['I', 'II', 'III', 'IV'], p=[0.35, 0.30, 0.25, 0.10])
        
        # Traitement reçu (peut être multiple)
        traitements = []
        if stade in ['I', 'II']:
            if np.random.random() < 0.8:
                traitements.append('Chirurgie')
            if np.random.random() < 0.6:
                traitements.append('Chimiothérapie')
        elif stade == 'III':
            if np.random.random() < 0.9:
                traitements.append('Chirurgie')
            if np.random.random() < 0.8:
                traitements.append('Chimiothérapie')
            if np.random.random() < 0.4:
                traitements.append('Radiothérapie')
        else:  # Stade IV
            if np.random.random() < 0.7:
                traitements.append('Chimiothérapie')
            if np.random.random() < 0.3:
                traitements.append('Radiothérapie')
            if np.random.random() < 0.2:
                traitements.append('Immunothérapie')
        
        traitement = '; '.join(traitements) if traitements else 'Surveillance'
        
        # Durée de séjour hospitalier (jours)
        if 'Chirurgie' in traitement:
            duree_sejour = int(np.random.gamma(2, 3) + 2)  # 2-15 jours, moyenne 8
        else:
            duree_sejour = int(np.random.exponential(3) + 1)  # 1-10 jours, moyenne 4
        duree_sejour = min(duree_sejour, 30)  # Maximum 30 jours
        
        # Statut vital (influence du stade et du type de cancer)
        prob_survie = 0.9 if stade in ['I', 'II'] else 0.7 if stade == 'III' else 0.5
        if type_cancer in ['Pancréas', 'Foie']:
            prob_survie *= 0.7  # Cancers plus agressifs
        elif type_cancer in ['Sein', 'Prostate']:
            prob_survie *= 1.1  # Meilleur pronostic
            
        statut_vital = 'Vivant' if np.random.random() < prob_survie else 'Décédé'
        
        # Comorbidités (plus fréquentes avec l'âge)
        comorbidites = []
        if age > 60 and np.random.random() < 0.3:
            comorbidites.append('Hypertension')
        if age > 65 and np.random.random() < 0.2:
            comorbidites.append('Diabète')
        if age > 70 and np.random.random() < 0.15:
            comorbidites.append('Maladie cardiovasculaire')
        if np.random.random() < 0.1:
            comorbidites.append('Maladie respiratoire')
            
        comorbidites_str = '; '.join(comorbidites) if comorbidites else 'Aucune'
        
        # Facteurs de risque
        tabac = 'Oui' if np.random.random() < 0.4 else 'Non'  # 40% de fumeurs
        if type_cancer == 'Poumon' and tabac == 'Oui':
            # Corrélation forte tabac-cancer poumon
            pass
        
        alcool = np.random.choice(['Jamais', 'Occasionnel', 'Régulier'], p=[0.4, 0.4, 0.2])
        
        # IMC (Indice de Masse Corporelle)
        imc = np.random.normal(25, 4)  # Moyenne 25, écart-type 4
        imc = max(min(imc, 45), 15)  # Limiter entre 15 et 45
        
        donnees.append({
            'patient_id': f'P{i+1:05d}',
            'age': age,
            'sexe': sexe,
            'region': region,
            'type_cancer': type_cancer,
            'stade': stade,
            'date_diagnostic': date_diagnostic,
            'traitement': traitement,
            'duree_sejour': duree_sejour,
            'statut_vital': statut_vital,
            'comorbidites': comorbidites_str,
            'tabac': tabac,
            'alcool': alcool,
            'imc': round(imc, 1)
        })
    
    return pd.DataFrame(donnees)

def nettoyer_donnees(df):
    """
    Nettoie et valide les données générées.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données brutes
        
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    
    print("Nettoyage des données...")
    
    # Convertir la date en format datetime
    df['date_diagnostic'] = pd.to_datetime(df['date_diagnostic'])
    
    # Créer des variables dérivées
    df['annee_diagnostic'] = df['date_diagnostic'].dt.year
    df['mois_diagnostic'] = df['date_diagnostic'].dt.month
    
    # Catégoriser l'âge
    df['groupe_age'] = pd.cut(
        df['age'], 
        bins=[0, 40, 50, 60, 70, 80, 100], 
        labels=['<40', '40-49', '50-59', '60-69', '70-79', '80+']
    )
    
    # Catégoriser l'IMC
    df['categorie_imc'] = pd.cut(
        df['imc'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Sous-poids', 'Normal', 'Surpoids', 'Obésité']
    )
    
    # Créer un indicateur de traitement multiple
    df['nb_traitements'] = df['traitement'].apply(
        lambda x: len(x.split(';')) if x != 'Surveillance' else 0
    )
    
    # Créer un score de risque simple
    score_risque = 0
    score_risque += (df['age'] > 65).astype(int) * 2
    score_risque += (df['stade'].isin(['III', 'IV'])).astype(int) * 3
    score_risque += (df['tabac'] == 'Oui').astype(int)
    score_risque += (df['comorbidites'] != 'Aucune').astype(int)
    
    df['score_risque'] = score_risque
    
    return df

def sauvegarder_donnees(df):
    """
    Sauvegarde les données dans différents formats.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
    """
    
    print("Sauvegarde des données...")
    
    # Données brutes
    df.to_csv('data/raw/cancer_data_france.csv', index=False, encoding='utf-8')
    
    # Données nettoyées
    df_clean = df.copy()
    df_clean.to_csv('data/processed/cancer_cleaned.csv', index=False, encoding='utf-8')
    
    # Format Excel pour faciliter l'exploration
    df_clean.to_excel('data/processed/cancer_data.xlsx', index=False)
    
    print("✅ Données sauvegardées avec succès !")
    print(f"   - Fichier CSV brut: data/raw/cancer_data_france.csv")
    print(f"   - Fichier CSV nettoyé: data/processed/cancer_cleaned.csv") 
    print(f"   - Fichier Excel: data/processed/cancer_data.xlsx")

def afficher_statistiques(df):
    """
    Affiche les statistiques descriptives des données générées.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    """
    
    print("\n" + "="*50)
    print("STATISTIQUES DES DONNÉES GÉNÉRÉES")
    print("="*50)
    
    print(f"Nombre total de patients: {len(df)}")
    print(f"Période couverte: {df['date_diagnostic'].min().strftime('%d/%m/%Y')} - {df['date_diagnostic'].max().strftime('%d/%m/%Y')}")
    
    print(f"\nRépartition par sexe:")
    print(df['sexe'].value_counts())
    
    print(f"\nÂge moyen: {df['age'].mean():.1f} ans")
    print(f"Âge médian: {df['age'].median():.1f} ans")
    
    print(f"\nTop 5 des types de cancer:")
    print(df['type_cancer'].value_counts().head())
    
    print(f"\nRépartition par stade:")
    print(df['stade'].value_counts())
    
    print(f"\nStatut vital:")
    print(df['statut_vital'].value_counts())
    
    print(f"\nDurée moyenne de séjour: {df['duree_sejour'].mean():.1f} jours")

def main():
    """
    Fonction principale qui orchestre la préparation des données.
    """
    
    print("🏥 PRÉPARATION DES DONNÉES CLINIQUES - CANCER FRANCE")
    print("="*60)
    
    # Génération des données
    df = generer_donnees_cancer()
    
    # Nettoyage
    df_clean = nettoyer_donnees(df)
    
    # Sauvegarde
    sauvegarder_donnees(df_clean)
    
    # Statistiques
    afficher_statistiques(df_clean)
    
    print(f"\n✨ Préparation terminée ! Vous pouvez maintenant analyser les données.")
    print(f"📁 Consultez le dossier 'data/' pour voir les fichiers générés.")

if __name__ == "__main__":
    main()
