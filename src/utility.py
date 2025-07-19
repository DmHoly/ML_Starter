# ---- Imports ----
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Optional, Type, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import plotly.graph_objects as go
import plotly.express as px

# ---- Type Aliases ----
FeatureTargetLink = Tuple[str, str, float, str]
FeatureFeatureLink = Tuple[str, str, float]

class ImportanceResultStore:
    """
    Stores all feature importances, including reliable and noise per feature.
    """
    def __init__(self):
        self.feature_target_links: List[Tuple[str, str, float, str]] = []
        self.feature_feature_links: List[Tuple[str, str, float]] = []
        # New: Keep detailed score for each feature
        self.feature_scores: Dict[str, Dict[str, float]] = {}  # {feature: {'reliable': ..., 'noise': ...}}
        self.random_thresholds: Dict[str, float] = {}

    def add_feature_target(self, src: str, tgt: str, imp: float, status: str):
        self.feature_target_links.append((src, tgt, imp, status))
        if src not in self.feature_scores:
            self.feature_scores[src] = {'reliable': 0.0, 'noise': 0.0}
        self.feature_scores[src][status] += imp

    def add_random_threshold(self, target: str, threshold: float):
        self.random_thresholds[target] = threshold

    def add_feature_feature(self, src: str, tgt: str, imp: float):
        self.feature_feature_links.append((src, tgt, imp))

    def get_feature_target_links(self, features=None, targets=None, include_noise=True):
        return [link for link in self.feature_target_links
                if (features is None or link[0] in features)
                and (targets is None or link[1] in targets)
                and (include_noise or link[3] != 'noise')]

    def get_feature_feature_links(self, features=None):
        return [link for link in self.feature_feature_links
                if (features is None or (link[0] in features and link[1] in features))]

    def get_all_feature_scores(self):
        """
        Returns a DataFrame with all features and their reliable/noise/total importances.
        """
        rows = []
        for f, d in self.feature_scores.items():
            rows.append({'feature': f, 'reliable': d.get('reliable', 0.0), 'noise': d.get('noise', 0.0),
                         'total': d.get('reliable', 0.0) + d.get('noise', 0.0)})
        return pd.DataFrame(rows)

    def get_average_random_threshold(self):
        if not self.random_thresholds:
            return 0.0
        return float(np.mean(list(self.random_thresholds.values())))

class FeatureImportanceCalc(BaseEstimator, TransformerMixin):
    """
    Calculates feature importances (reliable + noise) for both feature-to-target and feature-to-feature.
    """
    def __init__(
        self,
        model_target_cls=RandomForestRegressor,
        model_feature_cls=RandomForestRegressor,
        model_params=None,
        seed=42
    ):
        self.model_target_cls = model_target_cls
        self.model_feature_cls = model_feature_cls
        self.model_params = model_params or {}
        self.seed = seed

    def fit(self, X, y=None):
        Y = pd.DataFrame(y) if y is not None else pd.DataFrame()
        store = ImportanceResultStore()
        rng = np.random.RandomState(self.seed)
        Xc = pd.DataFrame(X).copy()
        Xc['_random'] = rng.rand(len(Xc))

        # Feature → Target
        for target in Y.columns:
            model = self.model_target_cls(random_state=self.seed, **self.model_params)
            model.fit(Xc, Y[target])
            imps = model.feature_importances_
            rand_imp = imps[list(Xc.columns).index('_random')]
            store.add_random_threshold(target, rand_imp)
            for feat, imp in zip(Xc.columns, imps):
                if feat == '_random':
                    continue
                status = 'reliable' if imp > rand_imp else 'noise'
                store.add_feature_target(feat, target, imp, status)

        # Feature → Feature
        for feat in X.columns:
            X_other = Xc.drop(columns=[feat])
            model = self.model_feature_cls(random_state=self.seed, **self.model_params)
            model.fit(X_other, Xc[feat])
            imps = model.feature_importances_
            rand_imp = imps[list(X_other.columns).index('_random')]
            for f, imp in zip(X_other.columns, imps):
                if f == '_random':
                    continue
                if imp > rand_imp:
                    store.add_feature_feature(f, feat, imp)

        self.store_ = store
        return self

    def transform(self, X):
        return pd.DataFrame(X)

    def get_store(self):
        return self.store_

class GraphAnalyzerEngine:
    """
    Orchestrates the analysis of feature importances and stores the results for visualization.
    """
    def __init__(
        self,
        model_target_cls: Type = RandomForestRegressor,
        model_feature_cls: Type = RandomForestRegressor,
        model_params: Optional[Dict] = None,
        seed: int = 42
    ):
        """
        Initializes the graph analyzer with model classes and parameters.
        """
        self.calc = FeatureImportanceCalc(
            model_target_cls=model_target_cls,
            model_feature_cls=model_feature_cls,
            model_params=model_params,
            seed=seed
        )
        self._store: Optional[ImportanceResultStore] = None

    def analyze(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Runs the feature importance analysis and stores the results.
        """
        self.calc.fit(X, Y)
        self._store = self.calc.get_store()

    def get_store(self) -> ImportanceResultStore:
        """
        Returns the stored importance results. Raises an error if analysis has not been run.
        """
        if self._store is None:
            raise RuntimeError('Call analyze() first.')
        return self._store

class VisualizerFactory:
    """
    Factory for creating various visualizations (Sankey, radar, bar, t-SNE) from importance results.
    """
    @staticmethod
    def make_sankey(
        store: 'ImportanceResultStore',
        features_to_display: Optional[List[str]] = None,
        targets_to_display: Optional[List[str]] = None,
        show_noise: bool = True,
        show_feature_feature_links: bool = True
    ) -> 'go.Figure':
        """
        Generate a Sankey diagram from feature importance results.
        If show_feature_feature_links is False, show only feature→target links.
        """
        ft_links = store.get_feature_target_links(
            features=features_to_display,
            targets=targets_to_display,
            include_noise=show_noise
        )
        if show_feature_feature_links:
            ff_links = store.get_feature_feature_links(features=features_to_display)
        else:
            ff_links = []

        links, colors = [], []
        for src, tgt, val, status in ft_links:
            links.append((src, tgt, val))
            colors.append('rgba(100,200,255,0.6)' if status == 'reliable' else 'rgba(200,200,200,0.2)')
        for src, tgt, val in ff_links:
            links.append((src, tgt, val))
            colors.append('rgba(150,150,150,0.3)')

        all_nodes = list({n for s, t, _ in links for n in (s, t)})
        idx = {n: i for i, n in enumerate(all_nodes)}

        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=all_nodes, pad=15, thickness=20, line=dict(color='black', width=0.5)),
            link=dict(
                source=[idx[s] for s, t, v in links],
                target=[idx[t] for s, t, v in links],
                value=[v for s, t, v in links],
                color=colors
            )
        )])
        fig.update_layout(title_text='Sankey Diagram', font_size=12)
        return fig

    @staticmethod
    def make_radar(store: ImportanceResultStore) -> go.Figure:
        """
        Generate a radar plot of aggregated feature importances.
        """
        scores = store.get_feature_scores()
        df = pd.DataFrame({'feature': list(scores.keys()), 'score': list(scores.values())})
        df = df.sort_values('score', ascending=False)
        categories = df['feature'].tolist() + [df['feature'].tolist()[0]]
        values = df['score'].tolist() + [df['score'].tolist()[0]]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title='Radar Plot')
        return fig

    @staticmethod
    def make_bar(store: ImportanceResultStore, show_threshold: bool = True, show_noise: bool = True) -> go.Figure:
        """
        Barplot des importances fiables + bruit (stacked) pour chaque feature.
        """
        df = store.get_all_feature_scores().sort_values('total', ascending=False)
        y_cols = ['reliable', 'noise'] if show_noise else ['reliable']
        fig = px.bar(df, x='feature', y=y_cols, text_auto='.2f',
                    title="Feature Importances (reliable + noise)" if show_noise else "Feature Importances (reliable only)")
        fig.update_layout(barmode='stack', xaxis_tickangle=-45)
        if show_threshold:
            threshold = store.get_average_random_threshold()
            fig.add_hline(y=threshold, line_dash='dash', annotation_text='Random Threshold')
        return fig
    
    @staticmethod
    def make_tsne(
        X,
        y=None,
        label_map=None,
        perplexity=30,
        random_state=42,
        title="t-SNE (all features in hover)"
    ):
        """
        Interactive t-SNE visualization. Hover shows all features for each point.
        - X: DataFrame (final features)
        - y: labels (optional)
        - label_map: mapping for y (optional)
        """
        import pandas as pd
        import plotly.graph_objects as go
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        X_proj = tsne.fit_transform(X)
        df = pd.DataFrame(X_proj, columns=["tsne1", "tsne2"], index=X.index)
        for col in X.columns:
            df[col] = X[col].values
        if y is not None:
            if label_map is not None:
                df["label"] = [label_map.get(label, label) for label in y]
            else:
                df["label"] = y
            color = df["label"]
        else:
            color = None
        if color is not None:
            fig = go.Figure(
                data=go.Scattergl(
                    x=df["tsne1"],
                    y=df["tsne2"],
                    mode="markers",
                    marker=dict(
                        color=pd.Categorical(color).codes,
                        colorscale="Viridis",
                        showscale=True,
                        size=8,
                        opacity=0.8
                    ),
                    text=df["label"] if "label" in df else None,
                    customdata=df[X.columns],
                    hovertemplate="<br>".join(
                        [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(X.columns)]
                    ) + "<br>label: %{text}<extra></extra>"
                )
            )
        else:
            fig = go.Figure(
                data=go.Scattergl(
                    x=df["tsne1"],
                    y=df["tsne2"],
                    mode="markers",
                    marker=dict(size=8, opacity=0.8),
                    customdata=df[X.columns],
                    hovertemplate="<br>".join(
                        [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(X.columns)]
                    ) + "<extra></extra>"
                )
            )
        fig.update_layout(title=title, xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
        return fig

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans and preprocesses data, handling missing values, scaling, encoding, and duplicate removal.
    """
    def __init__(
        self,
        numeric_strategy='mean',
        categorical_strategy='most_frequent',
        scale_numeric=True,
        drop_duplicates=False
    ):
        """
        Initializes the data cleaner with strategies for numeric and categorical features.
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scale_numeric = scale_numeric
        self.drop_duplicates = drop_duplicates

    def fit(self, X, y=None):
        """
        Fits the cleaning pipeline to the data, learning feature types and transformations.
        """
        X = pd.DataFrame(X)
        self.numeric_features_ = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if self.numeric_strategy == "drop" or self.categorical_strategy == "drop":
            self.pipeline_ = None
            return self
        
        transformers = []
        if self.numeric_features_:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.numeric_strategy)),
                ('scaler', StandardScaler() if self.scale_numeric else 'passthrough')
            ])
            transformers.append(('num', numeric_pipeline, self.numeric_features_))
        if self.categorical_features_:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.categorical_strategy)),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_features_))
        self.pipeline_ = ColumnTransformer(transformers, remainder='drop')
        self.pipeline_.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted cleaning pipeline.
        """
        X = pd.DataFrame(X)
        # Drop rows with NaNs if drop strategy is used
        if self.numeric_strategy == "drop" or self.categorical_strategy == "drop":
            X = X.dropna()
            return X.reset_index(drop=True)
        
        arr = self.pipeline_.transform(X)
        feature_names = []
        feature_names += self.numeric_features_
        if self.categorical_features_:
            try:
                enc = self.pipeline_.named_transformers_['cat'].named_steps['encoder']
                cats = enc.get_feature_names_out(self.categorical_features_)
                feature_names += cats.tolist()
            except Exception:
                pass
        return pd.DataFrame(arr, columns=feature_names, index=X.index)

class FeatureEngineeringSelector(BaseEstimator, TransformerMixin):
    """
    Generic feature engineering/selection pipeline supporting model-based selection, PCA, and custom strategies.
    """
    def __init__(self, strategies=None, problem_type='auto', random_state=42):
        """
        Initializes the feature engineering selector with strategies and problem type.
        """
        self.strategies = strategies or []
        self.problem_type = problem_type  # 'regression', 'classification', or 'auto'
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits the feature engineering/selection pipeline according to the specified strategies.
        """
        X = pd.DataFrame(X)
        self.features_ = X.columns.tolist()
        X_curr = X.copy()
        self.history_ = []
        y_series = pd.Series(y) if y is not None else None

        for strat in self.strategies:
            name = strat.get("name")
            if name == "model_importance":
                model_cls = strat.get("model_cls", RandomForestRegressor)
                model = model_cls(random_state=self.random_state)
                model.fit(X_curr, y_series)
                importances = np.array(model.feature_importances_)
                threshold = strat.get("threshold", "mean")
                if threshold == "mean":
                    thresh_val = np.mean(importances)
                elif threshold == "median":
                    thresh_val = np.median(importances)
                elif isinstance(threshold, float):
                    thresh_val = threshold
                else:
                    thresh_val = 0.0
                support = importances > thresh_val
                selected = X_curr.columns[support].tolist()
                X_curr = X_curr[selected]
                self.history_.append({
                    "step": "model_importance",
                    "selected": selected,
                    "importances": importances.tolist(),
                    "columns": X_curr.columns.tolist()
                })
            elif name == "pca":
                n_components = strat.get("n_components", 2)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = pca.fit_transform(X_curr)
                cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
                X_curr = pd.DataFrame(X_pca, columns=cols, index=X_curr.index)
                self.history_.append({
                    "step": "pca",
                    "columns": X_curr.columns.tolist(),
                    "explained_variance": pca.explained_variance_ratio_.tolist()
                })
            elif name == "custom":
                func = strat.get("func")
                X_curr = func(X_curr, y_series)
                self.history_.append({"step": "custom", "columns": X_curr.columns.tolist()})
            # Add other strategies here
        self.selected_features_ = X_curr.columns.tolist()
        self.X_shape_ = X_curr.shape
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted feature engineering/selection pipeline.
        """
        X = pd.DataFrame(X)
        if hasattr(self, "selected_features_"):
            cols = [c for c in self.selected_features_ if c in X.columns]
            return X[cols]
        else:
            return X

    def get_support(self):
        """
        Returns the list of selected features after feature engineering/selection.
        """
        return getattr(self, "selected_features_", [])

    def get_history(self):
        """
        Returns the history of feature engineering/selection steps.
        """
        return getattr(self, "history_", [])

class TargetLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes and decodes target labels using sklearn's LabelEncoder.
    """
    def __init__(self):
        """
        Initializes the target label encoder.
        """
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
    def fit(self, y):
        """
        Fits the label encoder to the target labels.
        """
        self._le.fit(y)
        return self
    def transform(self, y):
        """
        Transforms the target labels to encoded values.
        """
        return self._le.transform(y)
    def fit_transform(self, y):
        """
        Fits the label encoder and transforms the target labels.
        """
        return self._le.fit_transform(y)
    def inverse_transform(self, y):
        """
        Decodes the encoded target labels back to original labels.
        """
        return self._le.inverse_transform(y)




class MetaCleanPipeline:
    """
    Pipeline for cleaning, feature engineering, and target encoding.
    """
    def __init__(
        self,
        feature_engineering_strategies=None,
        drop_duplicates=True,
        target_encoder_method='label',  # 'label', 'ordinal', 'onehot', or a custom encoder class
        target_encoder_kwargs=None,
        auto_encode_features=True,
        **cleaner_kwargs
    ):
        self.cleaner = DataCleaner(drop_duplicates=drop_duplicates, **cleaner_kwargs)
        self.feat_eng = FeatureEngineeringSelector(strategies=feature_engineering_strategies) if feature_engineering_strategies else None
        self.drop_duplicates = drop_duplicates

        encoder_methods = {
            'label': LabelEncoder,
            'ordinal': OrdinalEncoder,
            'onehot': OneHotEncoder
        }
        
        self.target_encoder_kwargs = target_encoder_kwargs or {}
        if callable(target_encoder_method):
            self.target_encoder = target_encoder_method(**self.target_encoder_kwargs)
        else:
            encoder_cls = encoder_methods.get(target_encoder_method.lower(), LabelEncoder)
            self.target_encoder = encoder_cls(**self.target_encoder_kwargs)

        self.auto_encode_features = auto_encode_features

    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        df['__target__'] = y
        
        if self.drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        y_clean = df.pop('__target__')
        X_clean = df

        # Encode target
        if hasattr(self.target_encoder, 'fit_transform'):
            y_enc = self.target_encoder.fit_transform(y_clean.values.reshape(-1, 1) if isinstance(self.target_encoder, (OrdinalEncoder, OneHotEncoder)) else y_clean)
        else:
            self.target_encoder.fit(y_clean)
            y_enc = self.target_encoder.transform(y_clean)

        self._y_clean = y_enc
        self._X_clean = X_clean.copy()

        # Encode categorical features
        if self.auto_encode_features:
            cat_features = X_clean.select_dtypes(include=['object', 'category']).columns
            if cat_features.any():
                X_clean[cat_features] = X_clean[cat_features].astype(str)
                X_clean = pd.get_dummies(X_clean, columns=cat_features, drop_first=True)

        # Cleaning
        self.cleaner.fit(X_clean, y_enc)
        X_trans = self.cleaner.transform(X_clean)

        # Feature engineering
        if self.feat_eng:
            self.feat_eng.fit(X_trans, y_enc)

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if self.auto_encode_features:
            cat_features = X.select_dtypes(include=['object', 'category']).columns
            if cat_features.any():
                X[cat_features] = X[cat_features].astype(str)
                X = pd.get_dummies(X, columns=cat_features, drop_first=True)

        X_cleaned = self.cleaner.transform(X)

        if self.feat_eng:
            return self.feat_eng.transform(X_cleaned)
        
        return X_cleaned

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(self._X_clean)

    def transform_target(self, y):
        if hasattr(self.target_encoder, 'transform'):
            return self.target_encoder.transform(y.values.reshape(-1, 1) if isinstance(self.target_encoder, (OrdinalEncoder, OneHotEncoder)) else y)
        else:
            raise AttributeError("Target encoder does not support transform method.")

    def inverse_transform_target(self, y_enc):
        if hasattr(self.target_encoder, 'inverse_transform'):
            return self.target_encoder.inverse_transform(y_enc)
        else:
            raise AttributeError("Target encoder does not support inverse_transform method.")

    def get_cleaned_dataset(self):
        return self._X_clean.copy(), self._y_clean.copy()

    def get_final_dataset(self):
        return self.transform(self._X_clean)

    def get_selected_features(self):
        if self.feat_eng:
            return self.feat_eng.get_support()
        else:
            return self.cleaner.numeric_features_ + self.cleaner.categorical_features_

    def get_target_encoder(self):
        return self.target_encoder


#----------------------------------------------------
# Relabeling function for store
#----------------------------------------------------
def relabel_targets_in_store(store, label_map):
    """
    Replace encoded target names in the store by their human-readable names using the provided label map.
    """
    # Update feature_target_links
    new_links = []
    for src, tgt, imp, status in store.feature_target_links:
        try:
            new_tgt = label_map[int(tgt)]
        except (ValueError, KeyError):
            new_tgt = tgt
        new_links.append((src, new_tgt, imp, status))
    store.feature_target_links = new_links
    # Update random_thresholds (optional)
    new_thresholds = {}
    for tgt, val in store.random_thresholds.items():
        try:
            new_tgt = label_map[int(tgt)]
        except (ValueError, KeyError):
            new_tgt = tgt
        new_thresholds[new_tgt] = val
    store.random_thresholds = new_thresholds
    return store
