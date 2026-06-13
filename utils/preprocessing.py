import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from scipy.stats import chi2_contingency

class SurvivalDataProcessor:
    def __init__(self, data, global_survie=True, exclude_cols=None):
        # 1. Copie et nettoyage des espaces (Strip)
        # On le fait avant tout pour que les noms de colonnes et les valeurs soient propres
        self.data = data.copy().map(lambda x: x.strip() if isinstance(x, str) else x)
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(" ", "_") 
        self.data_original = self.data.copy()
        self.exclude_cols = exclude_cols or []

        # 2. Définition des cibles et mapping
        if global_survie:
            target_months = "overall_survival_(months)"
            target_status = "overall_survival_status"
            to_drop = ["relapse_free_status_(months)"]
        else:
            target_months = "relapse_free_status_(months)"
            target_status = "relapse_free_status"
            to_drop = ["overall_survival_(months)", "overall_survival_status"]

        # 3. Suppression des lignes sans durée (indispensable pour sksurv)
        self.data.dropna(subset=[target_months], inplace=True)
        self.data = self.data.drop(columns=to_drop)

        # 4. Encodage du statut (Mapping flexible)
        # On gère les deux formats classiques : Living/Deceased ou 0/1
        status_map = {"Living": False, "Deceased": True, "0": False, "1": True, 0: False, 1: True}
        self.data[target_status] = self.data[target_status].map(status_map)

        # 5. Renommage et nettoyage des colonnes inutiles
        self.data = self.data.rename(columns={
            target_months: "duration",
            target_status: "event"
        })
        self.data = self.data.drop(columns=self.exclude_cols)

        #C’est juste un identifiant de groupe (batch effect)
        self.data["cohort"] = self.data["cohort"].astype("category")
        self.data["tumor_stage"] = self.data["tumor_stage"].astype("category")       
        self.data["neoplasm_histologic_grade"] = self.data["neoplasm_histologic_grade"].astype("category")      

        # 6. Initialisation des listes
        self.numerical_features =  self.data.select_dtypes(exclude=['object', 'category']).columns 
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns
        self.imputer = None

    def get_data(self):
        """Retourne le DataFrame interne."""
        return self.data, self.data_original
    
    def get_features(self):
        """Retourne les listes des features numériques et catégorielles."""
        return self.numerical_features.tolist(), self.categorical_features.tolist()

    def impute_numerical(self, input_num = None,  method="MICE", n_neighbors=5):
        """Impute les valeurs manquantes sur le DataFrame interne."""
        if input_num is None:
            X_num = self.data[self.numerical_features].dropna(subset=["duration"])
        else:
            X_num = input_num

        # On s'assure que les données sont bien au format flottant pour l'imputer
        X_num = X_num.apply(pd.to_numeric, errors='coerce')
        
        if method == "MICE":
            self.imputer = IterativeImputer(random_state=42)
        else:
            self.imputer = KNNImputer(n_neighbors=n_neighbors)
            
        X_imputed = self.imputer.fit_transform(X_num)

        if input_num is not None:
            return X_imputed
        
        self.data[self.numerical_features] = X_imputed
        return self.data

    def get_chi2_significance(self):
        """Calcule la p-value du Chi2 face à la colonne 'event'."""
        scores = {}
        # On s'assure que 'event' ne contient pas de NaN avant le Chi2
        temp_data = self.data.dropna(subset=["event"])
        
        for col in self.categorical_features:
            if col in temp_data.columns:
                contingency = pd.crosstab(temp_data[col], temp_data["event"])
                _, p, _, _ = chi2_contingency(contingency)
                scores[col] = p
        
        return pd.Series(scores).sort_values()

    def get_features_summary(self, data):
        """Retourne un résumé rapide des données (manquants + types)."""
        return data[self.numerical_features + self.categorical_features].info()
    