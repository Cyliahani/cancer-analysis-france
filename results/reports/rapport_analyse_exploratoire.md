
# RAPPORT D'ANALYSE EXPLORATOIRE
## Données Cliniques - Cancer en France
### Généré le 17/09/2025 à 22:52

---

## RÉSUMÉ EXÉCUTIF

Cette analyse porte sur 5000 patients diagnostiqués avec un cancer entre 2020 et 2025.

### Points Clés:
- **Âge moyen**: 84.4 ans
- **Taux de survie global**: 82.7%
- **Cancer le plus fréquent**: Poumon
- **Durée moyenne de séjour**: 6.5 jours

---

## DONNÉES DÉMOGRAPHIQUES

### Répartition par sexe:
sexe
M    2654
F    2346

### Répartition par groupe d'âge:
groupe_age
40-49      48
50-59     233
60-69     556
70-79     804
80+      3358
<40         1

---

## TYPES DE CANCER

### Top 10 des cancers les plus fréquents:
type_cancer
Poumon        1051
Sein           812
Colorectal     770
Prostate       659
Vessie         420
Autres         365
Rein           300
Ovaire         186
Utérus         160
Foie           144

### Répartition par stade:
stade
I      1802
II     1467
III    1271
IV      460

---

## FACTEURS DE RISQUE

### Tabac:
tabac
Non    3041
Oui    1959

### Alcool:
alcool
Jamais         2047
Occasionnel    1977
Régulier        976

### IMC:
categorie_imc
Normal        2213
Surpoids      1966
Obésité        534
Sous-poids     287

---

## PRONOSTIC

### Survie par stade:
stade
I      91.1%
II     91.3%
III    71.2%
IV     53.7%

### Survie par type de cancer (Top 5):
type_cancer
Sein          90.4%
Prostate      90.0%
Poumon        82.3%
Vessie        81.0%
Colorectal    77.7%

---

## CORRÉLATIONS

Matrice de corrélation des variables numériques:
                  age  duree_sejour  nb_traitements  score_risque    imc
age             1.000        -0.032          -0.028         0.283  0.002
duree_sejour   -0.032         1.000           0.209        -0.055  0.007
nb_traitements -0.028         0.209           1.000         0.231 -0.002
score_risque    0.283        -0.055           0.231         1.000 -0.019
imc             0.002         0.007          -0.002        -0.019  1.000

---

## RECOMMANDATIONS

1. **Dépistage précoce**: 65.4% des cancers sont diagnostiqués aux stades I-II
2. **Facteurs de risque**: 39.2% des patients sont fumeurs
3. **Prise en charge**: Durée moyenne de séjour de 6.5 jours

---

*Rapport généré automatiquement par le script d'analyse exploratoire*
