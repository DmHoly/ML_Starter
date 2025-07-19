# ---- Imports ----
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Optional, Type, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


from itertools import combinations

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

class FeatureEngineeringSelector(BaseEstimator, TransformerMixin):
    """
    Generic feature engineering/selection pipeline supporting model-based selection, PCA, and custom strategies.
    """
    def __init__(self, strategies=None, problem_type='auto', random_state=42):
        """
        Initializes the feature engineering selector with strategies and problem type.
        strategies: list of dicts, each with keys:
            - name: "model_importance", "pca", or "custom"
            - other keys depending on strategy
        problem_type: 'regression', 'classification', or 'auto'
        random_state: random seed
        """
        self.strategies = strategies or []
        self.problem_type = problem_type
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits the feature engineering/selection pipeline according to the specified strategies.
        """
        X = pd.DataFrame(X)
        y_series = pd.Series(y) if y is not None else None

        # To store history and fitted parameters for transform
        self.history_ = []
        self.fitted_strategies_ = []

        X_curr = X.copy()
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

                selected = X_curr.columns[importances > thresh_val].tolist()
                X_curr = X_curr[selected]

                # Record history and fitted parameters
                self.history_.append({
                    "step": "model_importance",
                    "selected": selected,
                    "importances": importances.tolist(),
                })
                self.fitted_strategies_.append(("model_importance", {"selected": selected}))

            elif name == "pca":
                n_components = strat.get("n_components", 2)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca.fit(X_curr)

                X_pca = pca.transform(X_curr)
                cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
                X_curr = pd.DataFrame(X_pca, columns=cols, index=X_curr.index)

                # Record history and fitted PCA
                self.history_.append({
                    "step": "pca",
                    "columns": cols,
                    "explained_variance": pca.explained_variance_ratio_.tolist()
                })
                self.fitted_strategies_.append(("pca", {"pca": pca}))

            elif name == "custom":
                func = strat.get("func")
                X_curr = func(X_curr, y_series)

                self.history_.append({
                    "step": "custom",
                    "columns": X_curr.columns.tolist()
                })
                self.fitted_strategies_.append(("custom", {"func": func}))

            else:
                raise ValueError(f"Unknown strategy name: {name}")

        # Final selected features and shape
        self.selected_features_ = X_curr.columns.tolist()
        self.X_shape_ = X_curr.shape
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted feature engineering/selection pipeline.
        """
        X_curr = pd.DataFrame(X)
        for name, params in getattr(self, 'fitted_strategies_', []):
            if name == "model_importance":
                selected = params.get("selected", [])
                X_curr = X_curr[selected]

            elif name == "pca":
                pca = params.get("pca")
                X_pca = pca.transform(X_curr)
                cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
                X_curr = pd.DataFrame(X_pca, columns=cols, index=X_curr.index)

            elif name == "custom":
                func = params.get("func")
                X_curr = func(X_curr, None)

            else:
                # In case of unknown strategy
                continue

        return X_curr

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

class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible transformer for preprocessing a pandas DataFrame:
      - convert boolean columns to integers
      - convert object columns to categorical dtype
      - encode categorical features (label or one-hot)
      - handle missing values (drop, fill, or none)
      - normalize numeric features (standard or min-max)
      - provide inverse_transform for label encoding and normalization
      - generate dataset overview statistics and track colinearity
    """
    def __init__(self,
                 fill_strategy='drop',    # 'drop', 'fill', or 'none'
                 fill_method=None,        # 'mean', 'median', 'mode', or None
                 fill_value=None,         # constant fill value if fill_method is None
                 group_by=None,           # column name(s) for group-wise imputation
                 encoding='label',        # 'label' or 'onehot'
                 normalization=None,      # None, 'standard', or 'minmax'
                 normalization_range=(0,1), # tuple (min, max) for minmax scaler
                 corr_threshold=0.8       # threshold for detecting colinearity
                ):
        self.fill_strategy = fill_strategy
        self.fill_method = fill_method
        self.fill_value = fill_value
        self.group_by = group_by
        self.encoding = encoding
        self.normalization = normalization
        self.normalization_range = normalization_range
        self.corr_threshold = corr_threshold

        # Attributes to be set during fit
        self.bool_cols_ = []
        self.obj_cols_ = []
        self.cat_cols_ = []
        self.num_cols_ = []
        self.label_encoders_ = {}
        self.fill_values_ = {}
        self.scaler_ = None

    def fit(self, X, y=None):
        df = X.copy()
        # Identify columns by dtype
        self.bool_cols_ = df.select_dtypes(include=['bool']).columns.tolist()
        self.obj_cols_ = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = df.select_dtypes(include=['category']).columns.tolist()
        self.cat_cols_ = self.obj_cols_ + cat_cols
        self.num_cols_ = df.select_dtypes(include='number').columns.tolist()

        # Convert object columns to category dtype
        for col in self.obj_cols_:
            df[col] = df[col].astype('category')

        # Prepare imputation values if needed
        if self.fill_strategy == 'fill':
            if self.group_by:
                grouped = df.groupby(self.group_by)
                for col in df.columns:
                    if col in (self.group_by or []):
                        continue
                    if self.fill_method:
                        if self.fill_method in ('mean', 'median'):
                            self.fill_values_[col] = grouped[col].transform(self.fill_method)
                        else:
                            modes = grouped[col].apply(lambda grp: grp.mode().iloc[0] if not grp.mode().empty else None)
                            self.fill_values_[col] = df[self.group_by].apply(
                                lambda row: modes[tuple(row)] if isinstance(row, (list, tuple)) else modes[row],
                                axis=1
                            )
                    else:
                        self.fill_values_[col] = self.fill_value
            else:
                if self.fill_method:
                    if self.fill_method in ('mean', 'median'):
                        for col in self.num_cols_:
                            self.fill_values_[col] = getattr(df[col], self.fill_method)()
                    else:
                        mode_row = df.mode().iloc[0]
                        for col in df.columns:
                            self.fill_values_[col] = mode_row[col]
                else:
                    for col in df.columns:
                        self.fill_values_[col] = self.fill_value

        # Fit label encoders if needed
        if self.encoding == 'label':
            for col in self.cat_cols_:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders_[col] = le

        # Fit normalization scaler
        if self.normalization:
            if self.normalization == 'standard':
                self.scaler_ = StandardScaler()
            else:
                self.scaler_ = MinMaxScaler(feature_range=self.normalization_range)
            self.scaler_.fit(df[self.num_cols_])

        return self

    def transform(self, X):
        df = X.copy()
        # Convert booleans to integers
        for col in self.bool_cols_:
            if col in df:
                df[col] = df[col].astype(int)
        # Convert object to category
        for col in self.obj_cols_:
            if col in df:
                df[col] = df[col].astype('category')

        # Handle missing values
        if self.fill_strategy == 'drop':
            df = df.dropna()
        elif self.fill_strategy == 'fill':
            for col, vals in self.fill_values_.items():
                if col in df:
                    df[col] = df[col].fillna(vals)
        # else 'none': do nothing

        # Encode categorical features
        if self.encoding == 'label':
            for col, le in self.label_encoders_.items():
                if col in df:
                    df[col] = le.transform(df[col].astype(str))
        elif self.encoding == 'onehot':
            df = pd.get_dummies(df, columns=self.cat_cols_, drop_first=False)

        # Normalize numeric features
        if self.normalization and not df.empty:
            df[self.num_cols_] = self.scaler_.transform(df[self.num_cols_])

        return df

    def inverse_transform(self, X):
        df = X.copy()
        # Reverse normalization first
        if self.normalization and self.scaler_:
            df[self.num_cols_] = self.scaler_.inverse_transform(df[self.num_cols_])
        # Reverse label encoding
        if self.encoding == 'label':
            for col, le in self.label_encoders_.items():
                if col in df:
                    df[col] = le.inverse_transform(df[col].astype(int))
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for one-hot encoding.
        """
        if self.encoding == 'onehot' and input_features is not None:
            return self.transform(input_features).columns.tolist()
        return input_features or []

    def dataset_overview(self, X, corr_threshold=None):
        """
        Generate summary statistics for a DataFrame:
          - variance of numeric columns
          - count of unique values per column
          - count of missing values per column
          - data type of each column
          - correlation matrix for numeric features
          - list of colinear feature pairs above a threshold
        """
        df = X.copy()
        # Basic stats
        stats = pd.DataFrame({
            'variance': df.var(numeric_only=True),
            'unique_values': df.nunique(),
            'nan_counts': df.isna().sum(),
            'dtype': df.dtypes
        })
        # Correlation matrix
        corr_matrix = df.select_dtypes(include=np.number).corr()
        # Determine threshold
        thresh = corr_threshold if corr_threshold is not None else self.corr_threshold
        # Find colinear pairs
        colinear_pairs = [
            (col1, col2, corr_matrix.loc[col1, col2])
            for col1, col2 in combinations(corr_matrix.columns, 2)
            if abs(corr_matrix.loc[col1, col2]) >= thresh
        ]
        return {
            'stats': stats,
            'correlation_matrix': corr_matrix,
            'colinear_pairs': colinear_pairs
        }

    def get_label_mapping(self):
        """
        Returns a mapping for each label-encoded column:
          { column_name: {encoded_value: original_label} }
        """
        mapping = {}
        for col, le in self.label_encoders_.items():
            mapping[col] = {i: label for i, label in enumerate(le.classes_)}
        return mapping


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
