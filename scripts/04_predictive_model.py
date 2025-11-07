#!/usr/bin/env python3
"""
Modélisation Prédictive des Données Cliniques - Cancer en France
==============================================================

Ce script développe des modèles prédictifs pour estimer le risque de décès
et classifier les types de cancer basé sur les caractéristiques des patients.

Auteur: Cylia HANI
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, accuracy_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)

def charger_donnees():
    """
    Charge et prépare les données pour la modélisation.
    
    Returns:
        pd.DataFrame: DataFrame avec les données cliniques
    """
    
    try:
        df = pd.read_csv('data/processed/cancer_cleaned.csv')
        df['date_diagnostic'] = pd.to_datetime(df['date_diagnostic'])
        print(f" Données chargées: {len(df)} patients")
        return df
    except FileNotFoundError:
        print(" Fichier de données introuvable. Exécutez d'abord 01_data_preparation.py")
        return None

def preparer_donnees_survie(df):
    """
    Prépare les données pour la prédiction de survie.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    
    Returns:
        tuple: (X, y) données préparées pour l'entraînement
    """
    
    print("🔧 Préparation des données pour la prédiction de survie...")
    
    # Variables explicatives
    features = ['age', 'sexe', 'type_cancer', 'stade', 'imc', 'tabac', 
               'alcool', 'nb_traitements', 'score_risque']
    
    # Variable cible (1 = Vivant, 0 = Décédé)
    df_model = df[features + ['statut_vital']].copy()
    df_model = df_model.dropna()
    
    # Encoder la variable cible
    df_model['survie'] = (df_model['statut_vital'] == 'Vivant').astype(int)
    
    X = df_model[features]
    y = df_model['survie']
    
    print(f"Dataset de survie: {len(X)} patients")
    print(f"Taux de survie: {y.mean()*100:.1f}%")
    
    return X, y

def preparer_donnees_cancer(df):
    """
    Prépare les données pour la classification des types de cancer.
    
    Args:
        df (pd.DataFrame): DataFrame des données cliniques
    
    Returns:
        tuple: (X, y) données préparées pour l'entraînement
    """
    
    print("🔧 Préparation des données pour la classification des cancers...")
    
    # Garder seulement les 5 cancers les plus fréquents pour simplifier
    top_cancers = df['type_cancer'].value_counts().head(5).index
    df_cancer = df[df['type_cancer'].isin(top_cancers)].copy()
    
    # Variables explicatives
    features = ['age', 'sexe', 'imc', 'tabac', 'alcool', 'region']
    
    X = df_cancer[features]
    y = df_cancer['type_cancer']
    
    print(f"Dataset cancer: {len(X)} patients")
    print(f"Types de cancer: {y.value_counts().to_dict()}")
    
    return X, y

def creer_pipeline_survie():
    """
    Crée le pipeline de préprocessing et modélisation pour la survie.
    
    Returns:
        dict: Dictionnaire des pipelines de différents modèles
    """
    
    # Preprocessing
    numeric_features = ['age', 'imc', 'nb_traitements', 'score_risque']
    categorical_features = ['sexe', 'type_cancer', 'stade', 'tabac', 'alcool']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Différents modèles à tester
    models = {
        'Logistic_Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random_Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Gradient_Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ])
    }
    
    return models

def entrainer_modeles_survie(X, y):
    """
    Entraîne et évalue les modèles de prédiction de survie.
    
    Args:
        X (pd.DataFrame): Variables explicatives
        y (pd.Series): Variable cible
    
    Returns:
        dict: Résultats des modèles
    """
    
    print("Entraînement des modèles de survie...")
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Créer les pipelines
    models = creer_pipeline_survie()
    results = {}
    
    for name, pipeline in models.items():
        print(f"  Entraînement {name}...")
        
        # Entraînement
        pipeline.fit(X_train, y_train)
        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Validation croisée
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"    Précision: {accuracy:.3f}")
        print(f"    AUC: {auc:.3f}")
        print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results

def optimiser_meilleur_modele(X, y, results):
    """
    Optimise les hyperparamètres du meilleur modèle.
    
    Args:
        X (pd.DataFrame): Variables explicatives
        y (pd.Series): Variable cible
        results (dict): Résultats des modèles précédents
    
    Returns:
        Pipeline: Modèle optimisé
    """
    
    print("Optimisation du meilleur modèle...")
    
    # Trouver le meilleur modèle basé sur l'AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    print(f"Meilleur modèle: {best_model_name}")
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    if best_model_name == 'Random_Forest':
        # Paramètres à optimiser pour Random Forest
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        base_pipeline = creer_pipeline_survie()['Random_Forest']
        
    elif best_model_name == 'Gradient_Boosting':
        # Paramètres à optimiser pour Gradient Boosting
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
        base_pipeline = creer_pipeline_survie()['Gradient_Boosting']
        
    else:
        # Pour les autres modèles, retourner le modèle de base
        return results[best_model_name]['model']
    
    # Grid Search
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, 
                              scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Score optimisé: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def analyser_importance_features(best_model, X):
    """
    Analyse l'importance des features du meilleur modèle.
    
    Args:
        best_model (Pipeline): Modèle entraîné
        X (pd.DataFrame): Variables explicatives
    
    Returns:
        pd.Series: Importance des features
    """
    
    print("Analyse de l'importance des variables...")
    
    # Récupérer les noms des features après preprocessing
    try:
        preprocessor = best_model.named_steps['preprocessor']
        feature_names = []
        
        # Features numériques
        numeric_features = ['age', 'imc', 'nb_traitements', 'score_risque']
        feature_names.extend(numeric_features)
        
        # Features catégorielles (après one-hot encoding)
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(['sexe', 'type_cancer', 'stade', 'tabac', 'alcool'])
            feature_names.extend(cat_features)
        
        # Importance des features (si le modèle le supporte)
        classifier = best_model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feature_importance = pd.Series(importances, index=feature_names[:len(importances)])
            feature_importance = feature_importance.sort_values(ascending=False)
            
            print("Top 10 des variables les plus importantes:")
            for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
                print(f"  {i:2d}. {feature:<25}: {importance:.4f}")
            
            return feature_importance
            
    except Exception as e:
        print(f"Impossible d'analyser l'importance des features: {e}")
        return None

def visualiser_resultats_survie(results, feature_importance=None):
    """
    Crée les visualisations des résultats de prédiction de survie.
    
    Args:
        results (dict): Résultats des modèles
        feature_importance (pd.Series): Importance des features
    """
    
    print(" Génération des visualisations des résultats...")
    
    # Créer les dossiers nécessaires
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Comparaison des performances des modèles
    ax1 = plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    aucs = [results[name]['auc'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, accuracies, width, label='Précision', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, aucs, width, label='AUC', alpha=0.8)
    
    ax1.set_xlabel('Modèles')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparaison des Performances des Modèles')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.replace('_', ' ') for name in model_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax1.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                f'{accuracies[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                f'{aucs[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Courbe ROC du meilleur modèle
    ax2 = plt.subplot(2, 3, 2)
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_result = results[best_model_name]
    
    fpr, tpr, _ = roc_curve(best_result['y_test'], best_result['y_pred_proba'])
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {best_result["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Ligne de chance')
    ax2.set_xlabel('Taux de Faux Positifs')
    ax2.set_ylabel('Taux de Vrais Positifs')
    ax2.set_title(f'Courbe ROC - {best_model_name.replace("_", " ")}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Courbe Précision-Rappel
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(best_result['y_test'], best_result['y_pred_proba'])
    ax3.plot(recall, precision, linewidth=2)
    ax3.set_xlabel('Rappel')
    ax3.set_ylabel('Précision')
    ax3.set_title('Courbe Précision-Rappel')
    ax3.grid(True, alpha=0.3)
    
    # 4. Matrice de confusion
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Décédé', 'Vivant'], yticklabels=['Décédé', 'Vivant'])
    ax4.set_xlabel('Prédictions')
    ax4.set_ylabel('Réalité')
    ax4.set_title('Matrice de Confusion')
    
    # 5. Validation croisée
    ax5 = plt.subplot(2, 3, 5)
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    bars = ax5.bar(x_pos, cv_means, yerr=cv_stds, alpha=0.8, capsize=5)
    ax5.set_xlabel('Modèles')
    ax5.set_ylabel('AUC (Validation Croisée)')
    ax5.set_title('Validation Croisée (5-fold)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([name.replace('_', ' ') for name in model_names], rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + cv_stds[i] + 0.005,
                f'{cv_means[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Importance des features (si disponible)
    if feature_importance is not None:
        ax6 = plt.subplot(2, 3, 6)
        top_features = feature_importance.head(10)
        bars = ax6.barh(range(len(top_features)), top_features.values, alpha=0.8)
        ax6.set_yticks(range(len(top_features)))
        ax6.set_yticklabels(top_features.index)
        ax6.set_xlabel('Importance')
        ax6.set_title('Top 10 - Importance des Variables')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Ajouter les valeurs
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax6.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.5, 0.5, 'Importance des features\nnon disponible', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Importance des Variables')
    
    plt.tight_layout()
    plt.savefig('results/figures/survival_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Graphique sauvegardé: survival_prediction_results.png")

def predire_nouveaux_patients(best_model, exemples_patients):
    """
    Fait des prédictions sur de nouveaux patients exemples.
    
    Args:
        best_model (Pipeline): Modèle entraîné
        exemples_patients (pd.DataFrame): Nouveaux patients à prédire
    
    Returns:
        pd.DataFrame: Prédictions avec probabilités
    """
    
    print("Prédictions sur de nouveaux patients...")
    
    # Prédictions
    predictions = best_model.predict(exemples_patients)
    probabilities = best_model.predict_proba(exemples_patients)
    
    # Créer un DataFrame des résultats
    resultats = exemples_patients.copy()
    resultats['prediction_survie'] = ['Vivant' if p == 1 else 'Décédé' for p in predictions]
    resultats['probabilite_survie'] = probabilities[:, 1]
    resultats['risque_deces'] = 1 - probabilities[:, 1]
    
    return resultats

def creer_exemples_patients():
    """
    Crée des exemples de nouveaux patients pour les prédictions.
    
    Returns:
        pd.DataFrame: Exemples de patients
    """
    
    exemples = pd.DataFrame({
        'age': [55, 70, 45, 80, 60],
        'sexe': ['M', 'F', 'F', 'M', 'M'],
        'type_cancer': ['Poumon', 'Sein', 'Colorectal', 'Prostate', 'Poumon'],
        'stade': ['II', 'I', 'III', 'II', 'IV'],
        'imc': [25.5, 22.8, 28.3, 24.1, 29.7],
        'tabac': ['Oui', 'Non', 'Non', 'Non', 'Oui'],
        'alcool': ['Régulier', 'Occasionnel', 'Jamais', 'Occasionnel', 'Régulier'],
        'nb_traitements': [2, 1, 3, 2, 2],
        'score_risque': [4, 2, 5, 3, 7]
    })
    
    return exemples

def sauvegarder_modele(best_model, model_name='best_survival_model'):
    """
    Sauvegarde le meilleur modèle.
    
    Args:
        best_model (Pipeline): Modèle à sauvegarder
        model_name (str): Nom du fichier de sauvegarde
    """
    
    filepath = f'results/models/{model_name}.joblib'
    joblib.dump(best_model, filepath)
    print(f" Modèle sauvegardé: {filepath}")

def generer_rapport_modelisation(results, best_model_name, predictions_exemples):
    """
    Génère un rapport détaillé de la modélisation.
    
    Args:
        results (dict): Résultats des modèles
        best_model_name (str): Nom du meilleur modèle
        predictions_exemples (pd.DataFrame): Prédictions sur les exemples
    """
    
    best_result = results[best_model_name]
    
    rapport = f"""
# RAPPORT DE MODÉLISATION PRÉDICTIVE
## Prédiction de Survie - Cancer en France
### Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}

---

## RÉSUMÉ EXÉCUTIF

### Objectif
Développer un modèle prédictif pour estimer la probabilité de survie des patients atteints de cancer
basé sur leurs caractéristiques cliniques et démographiques.

### Meilleur Modèle: {best_model_name.replace('_', ' ')}
- **Précision**: {best_result['accuracy']:.3f}
- **AUC**: {best_result['auc']:.3f}
- **Validation croisée (AUC)**: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}

---

## PERFORMANCE DES MODÈLES

| Modèle | Précision | AUC | CV AUC (±SD) |
|--------|-----------|-----|--------------|
{chr(10).join([f"| {name.replace('_', ' ')} | {results[name]['accuracy']:.3f} | {results[name]['auc']:.3f} | {results[name]['cv_mean']:.3f} ± {results[name]['cv_std']:.3f} |" for name in results.keys()])}

---

## MÉTRIQUES DÉTAILLÉES

### Matrice de Confusion (Meilleur Modèle)
{pd.crosstab(best_result['y_test'], best_result['y_pred'], 
             rownames=['Réalité'], colnames=['Prédiction'], margins=True).to_string()}

### Rapport de Classification
{classification_report(best_result['y_test'], best_result['y_pred'], 
                      target_names=['Décédé', 'Vivant'])}

---

## PRÉDICTIONS SUR NOUVEAUX PATIENTS

Exemples de prédictions pour évaluer le modèle :

{predictions_exemples[['age', 'sexe', 'type_cancer', 'stade', 'prediction_survie', 'probabilite_survie']].to_string(index=False)}

---

## INTERPRÉTATION CLINIQUE

### Points Forts
- Modèle capable de discriminer les patients à risque
- AUC > 0.7 indique une performance acceptable
- Validation croisée stable

### Limitations
- Données simulées (non issues de vrais patients)
- Variables explicatives limitées
- Pas de suivi temporel (survie à long terme)

### Recommandations
1. **Validation externe** sur de vraies données
2. **Enrichissement** avec plus de variables cliniques
3. **Suivi longitudinal** pour améliorer les prédictions
4. **Interprétabilité** avec des modèles explicables

---

## UTILISATION PRATIQUE

### Cas d'Usage
- **Aide à la décision clinique** : Identifier les patients à haut risque
- **Allocation des ressources** : Prioriser les soins intensifs
- **Conseil patient** : Information sur le pronostic (avec précautions)

### Précautions
- Ce modèle est à des fins pédagogiques uniquement
- Ne doit pas remplacer l'expertise médicale
- Validation nécessaire avant usage clinique

---

*Modèle sauvegardé dans results/models/ pour réutilisation future*
"""
    
    # Sauvegarder le rapport
    with open('results/reports/rapport_modelisation.md', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print(f" Rapport de modélisation sauvegardé: results/reports/rapport_modelisation.md")

def main():
    """
    Fonction principale qui orchestre la modélisation prédictive.
    """
    
    print("MODÉLISATION PRÉDICTIVE - SURVIE CANCER")
    print("="*50)
    
    # Charger les données
    df = charger_donnees()
    if df is None:
        return
    
    # Préparer les données pour la prédiction de survie
    X, y = preparer_donnees_survie(df)
    
    # Entraîner les modèles
    results = entrainer_modeles_survie(X, y)
    
    # Optimiser le meilleur modèle
    best_model = optimiser_meilleur_modele(X, y, results)
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    
    # Analyser l'importance des features
    feature_importance = analyser_importance_features(best_model, X)
    
    # Visualiser les résultats
    visualiser_resultats_survie(results, feature_importance)
    
    # Prédictions sur de nouveaux patients
    exemples = creer_exemples_patients()
    predictions_exemples = predire_nouveaux_patients(best_model, exemples)
    
    print("\n PRÉDICTIONS SUR NOUVEAUX PATIENTS:")
    print("="*45)
    for i, row in predictions_exemples.iterrows():
        print(f"Patient {i+1}: {row['sexe']}, {row['age']} ans, {row['type_cancer']} stade {row['stade']}")
        print(f"  → Prédiction: {row['prediction_survie']} (probabilité: {row['probabilite_survie']:.1%})")
        print()
    
    # Sauvegarder le modèle
    sauvegarder_modele(best_model)
    
    # Générer le rapport
    generer_rapport_modelisation(results, best_model_name, predictions_exemples)
    
    print(f" Modélisation terminée !")
    print(f" Meilleur modèle: {best_model_name.replace('_', ' ')}")
    print(f"AUC: {results[best_model_name]['auc']:.3f}")
    print(f" Consultez le dossier 'results/' pour tous les résultats.")

if __name__ == "__main__":
    main()
