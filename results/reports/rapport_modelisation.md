
# RAPPORT DE MODÉLISATION PRÉDICTIVE
## Prédiction de Survie - Cancer en France
### Généré le 17/09/2025 à 23:01

---

## RÉSUMÉ EXÉCUTIF

### Objectif
Développer un modèle prédictif pour estimer la probabilité de survie des patients atteints de cancer
basé sur leurs caractéristiques cliniques et démographiques.

### Meilleur Modèle: Logistic Regression
- **Précision**: 0.832
- **AUC**: 0.773
- **Validation croisée (AUC)**: 0.748 ± 0.013

---

## PERFORMANCE DES MODÈLES

| Modèle | Précision | AUC | CV AUC (±SD) |
|--------|-----------|-----|--------------|
| Logistic Regression | 0.832 | 0.773 | 0.748 ± 0.013 |
| Random Forest | 0.815 | 0.758 | 0.701 ± 0.017 |
| Gradient Boosting | 0.833 | 0.766 | 0.741 ± 0.016 |
| SVM | 0.827 | 0.691 | 0.641 ± 0.028 |

---

## MÉTRIQUES DÉTAILLÉES

### Matrice de Confusion (Meilleur Modèle)
Prédiction   0    1   All
Réalité                  
0           29  144   173
1           24  803   827
All         53  947  1000

### Rapport de Classification
              precision    recall  f1-score   support

      Décédé       0.55      0.17      0.26       173
      Vivant       0.85      0.97      0.91       827

    accuracy                           0.83      1000
   macro avg       0.70      0.57      0.58      1000
weighted avg       0.80      0.83      0.79      1000


---

## PRÉDICTIONS SUR NOUVEAUX PATIENTS

Exemples de prédictions pour évaluer le modèle :

 age sexe type_cancer stade prediction_survie  probabilite_survie
  55    M      Poumon    II            Vivant            0.875913
  70    F        Sein     I            Vivant            0.957727
  45    F  Colorectal   III            Vivant            0.591506
  80    M    Prostate    II            Vivant            0.955876
  60    M      Poumon    IV            Décédé            0.460561

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
