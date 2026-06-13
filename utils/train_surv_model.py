import pandas as pd
from sklearn.compose import ColumnTransformer
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer # nécessaire from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.impute import SimpleImputer, IterativeImputer 
from sklearn.model_selection import cross_val_score

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

def km_by_group(df, time_col, event_col, group_col):
    """ - group_col : nom de la colonne contenant les groupes (optionnel) 
        - risk_scores : vecteur de scores pour créer les groupes (optionnel) 
        - n_groups : nombre de groupes à créer (par défaut 2) 
    """
    print(f"\n===== Kaplan-Meier : {group_col} =====")

    # Convertir en numérique
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    # Supprimer les lignes invalides
    df_clean = df.dropna(subset=[time_col, event_col, group_col])
    
    groups = df_clean[group_col].unique()
    km = KaplanMeierFitter()

    plt.figure(figsize=(8,6))

    for g in groups:
        mask = df_clean[group_col] == g

        if mask.sum() == 0:
            print(f"⚠️ Groupe '{g}' ignoré (aucune donnée valide).")
            continue

        km.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=str(g))
        km.plot()

    plt.title(f"Courbes de survie selon {group_col}")
    plt.xlabel("Temps (mois)")
    plt.ylabel("Probabilité de survie")
    plt.grid(True)
    plt.show()

    # Log-rank test si 2 groupes
    if len(groups) == 2:
        g1, g2 = groups
        mask1 = df_clean[group_col] == g1
        mask2 = df_clean[group_col] == g2

        result = logrank_test(
            df_clean[time_col][mask1], df_clean[time_col][mask2],
            df_clean[event_col][mask1], df_clean[event_col][mask2]
        )
        print("\nLog-rank test :")
        if result.p_value < 0.05:
            print("Log-rank test p-value :",result)
            print("Le modèle discrimine bien les risques")
        else:
            print("Log-rank test p-value :",result)
            print("Le modèle discrimine bien les risques")
            #print("les courbes ne sont pas significativement différentes")

class SurvivalModelWrapper:
    def __init__(self, data, model, numerical_features=None, categorical_features=None, 
                ordinal_features={}, exclude_cols = None,  global_survie=True):

        # prepoocessing des données: Copie et nettoyage des espaces (Strip) on rend les noms de colonnes et les valeurs soient propres
        self.data = data.copy().map(lambda x: x.strip() if isinstance(x, str) else x)
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(" ", "_") 
        self.data_original = self.data.copy()
        self.exclude_cols = exclude_cols or []
        self.global_survie = global_survie
    
        self.model = model
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.categorical_ordinal = list(ordinal_features.values())
        self.ordinal_features = list(ordinal_features.keys()) 
        self.pipeline = None
        self.c_index_ = None

        self.preprocessing(self.global_survie)
    
    def preprocessing(self, global_survie=True):
        """Prépare les préprocesseurs pour les différentes types de variables."""

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

        # C’est juste un identifiant de groupe (batch effect)
        #for col in ["tumor_stage", "neoplasm_histologic_grade", "cellularity"]: 
        #   self.data[col] = self.data[col].astype(str).str.strip() 
        #   self.data[col] = self.data[col].astype("category")     

    def _build_pipeline(self):
        """Crée le pipeline de transformation et le modèle."""
        
        # Retirer les ordinales des nominales
        nominal_cols = [c for c in self.categorical_features if c not in self.ordinal_features]

        # ----------------------------- Préprocesseurs ----------------------------- #
        # OneHot pour nominales
        nominal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # OrdinalEncoder pour ordinales
        ordinal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=self.categorical_ordinal))
        ])

        # IterativeImputer + StandardScaler pour numériques
        numeric_transformer = Pipeline(steps=[
            ("imputer", IterativeImputer(random_state=42)),
            ("scaler", StandardScaler())
        ])

        # ----------------------------- ColumnTransformer ----------------------------- #
        preprocessor = ColumnTransformer(
            transformers=[
                ("nominal", nominal_transformer, nominal_cols),
                ("ordinal", ordinal_transformer, self.ordinal_features),
                ("numeric", numeric_transformer, self.numerical_features)
            ],
            remainder="drop"
        )

        # ----------------------------- Pipeline final ----------------------------- #
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("survival_model", self.model)
        ])

    def fit(self, event_col="event", duration_col="duration", test_size=0.1, random_state=42):
        """Prépare les données, entraîne le modèle et calcule le C-index de test."""
        # 1. Séparation X et y
        X = self.data[self.categorical_features + self.numerical_features + self.ordinal_features]
        y = self.data[[event_col, duration_col]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2. Conversion au format sksurv
        y_train_surv = Surv.from_dataframe(event_col, duration_col, y_train)
        y_test_surv = Surv.from_dataframe(event_col, duration_col, y_test)

        # 3. Construction et entraînement
        self._build_pipeline()
        self.pipeline.fit(X_train, y_train_surv)

        # 4. Évaluation
        risk_test = self.pipeline.predict(X_test)
        self.c_index_ = concordance_index_censored(
            y_test_surv[event_col], 
            y_test_surv[duration_col], 
            risk_test
        )[0]
        
        return X_test, y_test_surv, y_train_surv

    def predict_risk(self, X):
        """Prédit le score de risque pour de nouvelles données."""
        if self.pipeline is None:
            raise ValueError("Le modèle doit être entraîné avant de prédire.")
        return self.pipeline.predict(X)
    
    def get_pipeline(self):
        """Initialise et retourne le pipeline sans l'entraîner."""
        if self.pipeline is None:
            self._build_pipeline()
        return self.pipeline
    
    def get_feature(self):
        return [self.categorical_features + self.numerical_features + self.ordinal_features]
    
    def get_data(self):
        return self.data, self.data_original
    
    def tune_parameters(self, param_grid, event_col="event", duration_col="duration", cv=5):
        """Exécute un GridSearchCV et met à jour le pipeline avec les meilleurs paramètres."""
        X = self.data[self.categorical_features + self.numerical_features + self.ordinal_features]
        y = Surv.from_dataframe(event_col, duration_col, self.data[[event_col, duration_col]])
        
        if self.pipeline is None:
            self._build_pipeline()

        gcv = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv, n_jobs=-1)
        gcv.fit(X, y)
        
        # On met à jour le pipeline de la classe avec le meilleur modèle trouvé
        self.pipeline = gcv.best_estimator_
        self.c_index_ = gcv.best_score_
        
        print(f"Meilleurs paramètres : {gcv.best_params_}")
        return gcv.best_params_

    def evaluate_with_cv(self, cv=5, event_col="event", duration_col="duration"):
        """
        Évalue la performance du modèle en utilisant la validation croisée
        sur l'ensemble des données.
        """
        X = self.data[self.categorical_features + self.numerical_features + self.ordinal_features]
        y = Surv.from_dataframe(event_col, duration_col, self.data[[event_col, duration_col]])
        
        if self.pipeline is None:
            self._build_pipeline()

        # cross_val_score utilise par défaut le score du modèle. 
        # Pour sksurv, c'est le concordance index.
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        
        print(f"--- Résultats de la Validation Croisée ({cv} folds) ---")
        print(f"C-index moyen : {scores.mean():.4f}")
        print(f"Écart-type     : {scores.std():.4f}")
        print(f"Scores par fold : {scores}")
        
        return scores