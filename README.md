# Breast-Cancer-Survival-ML: Analyse de Survie et Stratification du Risque du Cancer du Sein

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Survival](https://img.shields.io/badge/Library-Scikit--Survival-orange.svg)](https://scikit-survival.readthedocs.io/)
[![Medical Data](https://img.shields.io/badge/Domain-Oncology%20/%20Healthcare-red.svg)](https://www.kaggle.com/datasets/balgopal/breast-cancer-metabric)

## 🎯 Objectif du Projet
Ce projet implémente un pipeline complet d'**analyse de survie** (Survival Analysis) basé sur le dataset clinique et moléculaire **METABRIC**. L'objectif est de prédire le risque de mortalité des patientes atteintes d'un cancer du sein et de les **stratifier en groupes de risque distincts** afin d'aider à la personnalisation des parcours de soins.

---

## 🚀 Fonctionnalités Clés & Pipeline

### 1. Préparation & Nettoyage Clinique (EDA avancé)
*   **Imputation MICE (IterativeImputer) :** Gestion robuste des données manquantes pour les variables numériques.
*   **Ingénierie des caractéristiques :** Modélisation des variables clés selon la logique oncologique (Index Pronostique de Nottingham (NPI), sous-types moléculaires PAM50, envahissement ganglionnaire).
*   **Contrôle des biais :** Exclusion des variables post-diagnostic ou redondantes pour éviter le *data leakage* et la multicolinéarité (évaluée via le V de Cramér).

### 2. Modélisation Prédictive (Benchmark)
Entraînement et optimisation par validation croisée (`GridSearchCV`) de 3 architectures de pointe en analyse de survie :
*   **Cox Proportional Hazards (CoxPH)** (Modèle linéaire de référence)
*   **Random Survival Forest (RSF)** (Modèle d'ensemble non-linéaire)
*   **Gradient Boosting Survival Analysis (GBSA)**

### 3. Évaluation & Validation Clinique
*   **Performance globale :** Évaluation via le **C-index** (Concordance Index).
*   **Calibration temporelle :** Calcul et affichage des courbes de **Brier Score** et de l'**IBS (Integrated Brier Score)**.
*   **Interprétabilité :** Extraction de l'importance des features via **Permutation Importance**.

### 4. Application Clinique : Stratification du Risque
*   Séparation automatique des patientes en deux cohortes (**Bas Risque vs Haut Risque**) basée sur le score de risque médian.
*   Génération des courbes de **Kaplan-Meier** et validation statistique de la séparation par le **Log-Rank Test**.

---

## 🛠️ Stack Technique
*   **Langage :** Python
*   **Manipulation de données :** `pandas`, `numpy`
*   **Machine Learning de Survie :** `scikit-survival`, `lifelines`
*   **Visualisation :** `matplotlib`, `seaborn`

---

## 📈 Structure des Résultats attendus

Le pipeline génère automatiquement les livrables suivants :
1.  **Comparatif des C-index** entre Cox, RSF et Gradient Boosting.
2.  **Graphique de Permutation Importance** identifiant les biomarqueurs les plus discriminants (ex: NPI, Stade de la tumeur).
3.  **Courbes de Kaplan-Meier** démontrant visuellement la capacité du modèle à isoler le groupe à haut risque pour une prise en charge agressive.

---

## 💻 Comment lancer le projet

1. **Cloner le projet**
```bash
git clone [https://github.com/VOTRE_PSEUDO/OncoStratify-METABRIC.git](https://github.com/VOTRE_PSEUDO/OncoStratify-METABRIC.git)
cd OncoStratify-METABRIC
```
2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```
2. Lancer le notebook ou le script principal
```bash
jupyter notebook breast_predict.ipynb
```
---

<FollowUp label="Tu veux que je t'aide à rédiger le fichier 'requirements.txt' correspondant à tes imports ?" query="Peux-tu me générer le contenu type du fichier requirements.txt pour ce projet d'analyse de survie METABRIC ?"/>
