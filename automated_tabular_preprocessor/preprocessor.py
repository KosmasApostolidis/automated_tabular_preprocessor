from __future__ import annotations

import pandas                   as pd
import numpy                    as np
from abc                        import ABC, abstractmethod
from sklearn.preprocessing      import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble           import ExtraTreesClassifier
from sklearn.feature_selection  import SelectKBest, f_classif
from collections                import Counter
from imblearn.over_sampling     import SMOTE
from ctgan                      import CTGAN, TVAE

class AbstractPreprocessor(ABC):
    def __init__(self,
                 df                       = None,
                 target_column            = None,
                 cols_to_drop             = None,
                 value_to_replace         = None,
                 missing_values_threshold = 0.4,
                 random_state             = 42,
                 **kwargs):

        self.df                       = df
        self.target_column            = target_column
        self.cols_to_drop             = cols_to_drop if cols_to_drop is not None else []
        self.value_to_replace         = value_to_replace
        self.missing_values_threshold = missing_values_threshold
        self.random_state             = random_state

        self.imputation_values        = {}
        self.mode_replacements        = {}
        self.scaler_object            = None
        self.columns_to_keep          = None
        self.cat_encoding_mappings    = {}
        self.high_missing_cols_dropped = []

    def _drop_duplicates(self):
        print("\n--- Dropping Duplicates ---")
        if self.df is None:
            print("DataFrame is None. Skipping.")
            return self
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f'Dropped {initial_len - len(self.df)} duplicates.')
        return self

    def _handle_missing_values(self, drop_high_missing=True, numerical_imputer='median', is_train=True):
        print(f"\n--- Handling Missing Values (Train={is_train}) ---")

        if not is_train and self.df.isnull().sum().sum() > 0:
            print("Warning: Missing values detected in Test set. Using training stats to fill.")

        missing_percentage = self.df.isnull().sum() / len(self.df)
        cols_with_missing  = missing_percentage[missing_percentage > 0].index.tolist()

        if is_train:
            if drop_high_missing:
                cols_to_drop = [c for c in cols_with_missing if missing_percentage[c] > self.missing_values_threshold]
                self.high_missing_cols_dropped = cols_to_drop
                if cols_to_drop:
                    self.df.drop(columns=cols_to_drop, inplace=True)
                    print(f"Dropped columns (> threshold): {cols_to_drop}")

            cols_to_impute = [c for c in self.df.columns if self.df[c].isnull().any()]
            for col in cols_to_impute:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].median() if numerical_imputer == 'median' else self.df[col].mean()
                    self.imputation_values[col] = val
                else:
                    if not self.df[col].mode().empty:
                        val = self.df[col].mode()[0]
                        self.imputation_values[col] = val
        else:
            if self.high_missing_cols_dropped:
                cols_to_drop = [c for c in self.high_missing_cols_dropped if c in self.df.columns]
                if cols_to_drop:
                    self.df.drop(columns=cols_to_drop, inplace=True)
                    print(f"Dropped columns (from training): {cols_to_drop}")

        for col, val in self.imputation_values.items():
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(val)
                print(f" - Filled '{col}' with {val}")

        return self

    def _drop_columns(self):
        print(f"\n--- Dropping User-Specified Columns: {self.cols_to_drop} ---")
        if self.df is None:
            print("DataFrame is None. Skipping drop columns.")
            return self
        if self.cols_to_drop:
            self.df.drop(columns=self.cols_to_drop, inplace=True, errors='ignore')
        else:
            print("No columns to drop.")
        print(f"New shape after dropping: {self.df.shape}")
        return self

    def _replace_value_with_mode(self, is_train=True):
        print(f"\n--- Replacing '{self.value_to_replace}' with Mode (Train={is_train}) ---")
        if self.value_to_replace is None:
            return self

        if is_train:
            for col in self.df.columns:
                if (self.df[col] == self.value_to_replace).any():
                    valid_data = self.df[self.df[col] != self.value_to_replace][col]
                    if not valid_data.empty:
                        self.mode_replacements[col] = valid_data.mode()[0]

        for col, mode_val in self.mode_replacements.items():
            if col in self.df.columns:
                mask = self.df[col] == self.value_to_replace
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, col] = mode_val
                    print(f" - Replaced {count} instances in '{col}' with '{mode_val}'")
        return self

    def _check_class_balance(self, y, title=""):
        print(f"\n--- Class Balance Check ({title}) ---")

        target_series = None
        if isinstance(y, pd.DataFrame):
            target_series = y[self.target_column]
        elif isinstance(y, pd.Series):
            target_series = y
        else:
            raise TypeError(f"Expected a pandas DataFrame or Series, but got {type(y)}")

        class_counts = Counter(target_series)
        print(f"Distribution for target '{self.target_column}':")
        for label, count in class_counts.items():
            percentage = (count / len(target_series)) * 100
            print(f"  Class {label}: {count} samples ({percentage:.2f}%)")

        return self

    @abstractmethod
    def run_preprocessing_pipeline(self):
        pass

class TabularDataPreprocessor(AbstractPreprocessor):
    def __init__(self,
                 number_of_top_k_features   = 0,
                 features_to_skip_scaling   = None,
                 numerical_scaler           = 'standard',
                 categorical_encoder        = 'onehot',
                 model_type                 = 'classical',
                 feature_selection_method   = 'trees',
                 iqr_multiplier             = 1.5,
                 remove_highly_correlated   = False,
                 remove_low_variance        = False,
                 remove_outliers            = False,
                 encode_categorical         = True,
                 **kwargs):

        super().__init__(**kwargs)
        self.number_of_top_k_features = number_of_top_k_features
        self.top_k_features_columns   = None
        self.features_to_skip_scaling = features_to_skip_scaling or []
        self.numerical_scaler         = numerical_scaler
        self.categorical_encoder      = categorical_encoder
        self.model_type               = model_type
        self.final_target_cols        = [self.target_column]
        self.feature_selection_method = feature_selection_method
        self.iqr_multiplier           = iqr_multiplier
        self.remove_highly_correlated = remove_highly_correlated
        self.remove_low_variance      = remove_low_variance
        self.remove_outliers          = remove_outliers
        self.encode_categorical       = encode_categorical
        self.encoded_categorical_cols = []
        self.target_encoding_mapping  = None
        self.target_was_encoded       = False
        self.highly_correlated_cols   = []
        self.low_variance_cols        = []

    def _encode_categorical_features(self, is_train=True):
        print(f"\n--- Encoding Categorical Features (Train={is_train}) ---")

        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in cat_cols:
            cat_cols.remove(self.target_column)

        if not cat_cols:
            return self

        if self.categorical_encoder == 'onehot':
            cols_before = set(self.df.columns)
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True, dtype=int)
            self.encoded_categorical_cols = [c for c in self.df.columns if c not in cols_before]

            if is_train:
                self.columns_after_encoding = self.df.columns.tolist()
            else:
                for col in self.columns_after_encoding:
                    if col not in self.df.columns:
                        self.df[col] = 0
                self.df = self.df[self.columns_after_encoding]

        elif self.categorical_encoder == 'factorize':
             self.encoded_categorical_cols = list(cat_cols)
             for col in cat_cols:
                if is_train:
                    codes, uniques = pd.factorize(self.df[col])
                    self.cat_encoding_mappings[col] = {val: i for i, val in enumerate(uniques)}
                    self.df[col] = codes
                else:
                    mapping = self.cat_encoding_mappings.get(col, {})
                    self.df[col] = self.df[col].map(mapping).fillna(-1).astype(int)

             if is_train:
                 self.columns_after_encoding = self.df.columns.tolist()

        return self

    def _encode_target_column(self, is_train=True):
        print(f"\n--- Encoding Target Column (Train={is_train}) ---")

        if is_train and self.df[self.target_column].dtype not in ['object', 'category']:
            print("Target column is already numeric. Skipping encoding.")
            return self

        if not is_train and not self.target_was_encoded:
            print("Target column is already numeric. Skipping encoding.")
            return self

        if self.model_type == 'neural_network':
            print(f"Applying ONE-HOT ENCODING to target '{self.target_column}' for Neural Network.")

            if is_train:
                original_cols = set(self.df.columns)
                self.df = pd.get_dummies(self.df, columns=[self.target_column], prefix=self.target_column, dtype=int)
                self.final_target_cols = sorted(set(self.df.columns) - original_cols)
                self.target_was_encoded = True
                print(f" - Target column replaced by: {self.final_target_cols}")
            else:
                original_cols = set(self.df.columns)
                self.df = pd.get_dummies(self.df, columns=[self.target_column], prefix=self.target_column, dtype=int)
                new_target_cols = list(set(self.df.columns) - original_cols)
                for col in self.final_target_cols:
                    if col not in self.df.columns:
                        self.df[col] = 0
                extra = [c for c in new_target_cols if c not in self.final_target_cols]
                if extra:
                    self.df.drop(columns=extra, inplace=True)

        else:
            if is_train:
                print(f"Applying LABEL ENCODING to target '{self.target_column}'.")
                self.df[self.target_column], uniques = pd.factorize(self.df[self.target_column])
                self.target_encoding_mapping = {val: i for i, val in enumerate(uniques)}
                self.target_was_encoded = True
                self.final_target_cols = [self.target_column]
                print(f" - Target mapping: {self.target_encoding_mapping}")
            else:
                print(f"Applying learned LABEL ENCODING to target '{self.target_column}'.")
                self.df[self.target_column] = self.df[self.target_column].map(
                    self.target_encoding_mapping
                ).fillna(-1).astype(int)

        return self

    def _remove_outliers_iqr(self):
        print(f"\n--- Removing Outliers using IQR (Multiplier: {self.iqr_multiplier}) ---")

        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)

        if hasattr(self, 'encoded_categorical_cols'):
            numerical_cols = [col for col in numerical_cols if col not in self.encoded_categorical_cols]

        if not numerical_cols:
            print("No numerical columns to process for outliers. Skipping.")
            return self

        initial_rows    = len(self.df)
        outlier_indices = set()

        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - (self.iqr_multiplier * IQR)
            upper_bound = Q3 + (self.iqr_multiplier * IQR)

            col_outlier_indices = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index
            outlier_indices.update(col_outlier_indices)

        if outlier_indices:
            self.df.drop(index=list(outlier_indices), inplace=True)

        rows_removed = initial_rows - len(self.df)
        print(f"Removed {rows_removed} outlier rows. New shape: {self.df.shape}")

        return self

    def _remove_highly_correlated_features(self, threshold=0.8, is_train=True):
        print(f"\n--- Removing Highly Correlated Features (Train={is_train}) ---")

        if is_train:
            numeric_df = self.df.select_dtypes(include=np.number)
            protected = set(self.final_target_cols) | {self.target_column}
            cols_to_exclude = [c for c in protected if c in numeric_df.columns]
            if cols_to_exclude:
                numeric_df = numeric_df.drop(columns=cols_to_exclude)
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.highly_correlated_cols = [column for column in upper.columns if any(upper[column] > threshold)]
            if self.highly_correlated_cols:
                self.df.drop(columns=self.highly_correlated_cols, inplace=True)
                print(f" - Dropped correlated columns: {self.highly_correlated_cols}")
        else:
            cols_to_drop = [c for c in self.highly_correlated_cols if c in self.df.columns]
            if cols_to_drop:
                self.df.drop(columns=cols_to_drop, inplace=True)
                print(f" - Dropped correlated columns (from training): {cols_to_drop}")

        return self

    def _remove_low_variance_features(self, threshold=0.01, is_train=True):
        print(f"\n--- Removing Low Variance Features (Train={is_train}) ---")

        if is_train:
            protected = set(self.final_target_cols) | {self.target_column}
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            candidates = [col for col in numeric_cols if col not in protected]
            self.low_variance_cols = [col for col in candidates if self.df[col].var() < threshold]
            if self.low_variance_cols:
                self.df.drop(columns=self.low_variance_cols, inplace=True)
                print(f" - Dropped low variance columns: {self.low_variance_cols}")
        else:
            cols_to_drop = [c for c in self.low_variance_cols if c in self.df.columns]
            if cols_to_drop:
                self.df.drop(columns=cols_to_drop, inplace=True)
                print(f" - Dropped low variance columns (from training): {cols_to_drop}")

        return self

    def _select_top_k_features(self, is_train=True):
        if self.number_of_top_k_features <= 0:
            return self

        target_cols = [c for c in self.final_target_cols if c in self.df.columns]

        if is_train:
            print(f"\n--- Selecting Top {self.number_of_top_k_features} Features ---")
            X = self.df.drop(columns=target_cols)
            y = self.df[target_cols[0]] if len(target_cols) == 1 else self.df[target_cols]

            if self.feature_selection_method == 'trees':
                model = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state)
                model.fit(X, y if isinstance(y, pd.Series) else y.values.argmax(axis=1))
                importances = pd.Series(model.feature_importances_, index=X.columns)
                self.columns_to_keep = importances.nlargest(self.number_of_top_k_features).index.tolist()

            elif self.feature_selection_method == 'f_classif':
                y_for_selection = y if isinstance(y, pd.Series) else y.values.argmax(axis=1)
                selector = SelectKBest(score_func=f_classif, k=self.number_of_top_k_features)
                selector.fit(X, y_for_selection)
                self.columns_to_keep = X.columns[selector.get_support()].tolist()

            self.columns_to_keep.extend(target_cols)

        if self.columns_to_keep:
            self.df = self.df[self.columns_to_keep]
            print(f"Features reduced to: {self.df.shape[1]}")

        return self

    def _scale_numerical_features(self, is_train=True):
        print(f"\n--- Scaling Numerical Features (Train={is_train}) ---")

        nums = self.df.select_dtypes(include=np.number).columns.tolist()
        protected = set(self.final_target_cols) | {self.target_column}
        cols_to_scale = [c for c in nums if c not in protected and c not in self.features_to_skip_scaling]

        if not cols_to_scale:
            return self

        if is_train:
            if self.numerical_scaler == 'standard': self.scaler_object = StandardScaler()
            elif self.numerical_scaler == 'minmax': self.scaler_object = MinMaxScaler()
            elif self.numerical_scaler == 'robust': self.scaler_object = RobustScaler()
            else:
                raise ValueError(
                    f"Unknown numerical_scaler '{self.numerical_scaler}'. "
                    f"Valid options: 'standard', 'minmax', 'robust'."
                )

            self.df[cols_to_scale] = self.scaler_object.fit_transform(self.df[cols_to_scale])
        else:
            if self.scaler_object:
                self.df[cols_to_scale] = self.scaler_object.transform(self.df[cols_to_scale])
        return self

    def run_preprocessing_pipeline(self, is_train=True):
        self._drop_columns()
        self._drop_duplicates()
        self._handle_missing_values(is_train=is_train)

        if self.value_to_replace is not None:
            self._replace_value_with_mode(is_train=is_train)

        if is_train and self.remove_outliers:
            self._remove_outliers_iqr()

        if self.encode_categorical:
            self._encode_categorical_features(is_train=is_train)

        self._encode_target_column(is_train=is_train)

        self._post_target_encoding_hook(is_train=is_train)

        if self.remove_highly_correlated:
            self._remove_highly_correlated_features(is_train=is_train)

        if self.remove_low_variance:
            self._remove_low_variance_features(is_train=is_train)

        self._select_top_k_features(is_train=is_train)

        self._scale_numerical_features(is_train=is_train)
        return self

    def _post_target_encoding_hook(self, is_train=True):
        """Hook for subclasses to inject steps after target encoding."""
        pass

class AugmentedDataPreprocessor(TabularDataPreprocessor):
    def __init__(self, *args, strategy='smote', **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        print(f"Initialized Augmented preprocessor with strategy: {self.strategy.upper()}.")

    def _post_target_encoding_hook(self, is_train=True):
        if is_train:
            self._augment_data()

    def _augment_data(self):
        print(f"\n--- Augmenting Data using {self.strategy.upper()} ---")
        if self.target_column is None:
            print("No target_column specified. Cannot perform augmentation.")
            return self
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        print("Class distribution before augmentation:")
        print(y.value_counts())

        if self.strategy == 'smote':
            if self.df.select_dtypes(include=['object', 'category']).shape[1] > 0:
                print("Error: SMOTE requires all features to be numeric. Ensure encoding runs first.")
                return self
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=self.target_column)], axis=1)
        elif self.strategy == 'ctgan' or self.strategy == 'tvae':
            synth = CTGAN(epochs=300) if self.strategy == 'ctgan' else TVAE(epochs=300)
            synth.fit(self.df, discrete_columns=[self.target_column])
            n_samples = len(self.df)
            synthetic = synth.sample(n_samples)
            self.df = pd.concat([self.df, synthetic], ignore_index=True)
        else:
            print(f"Unknown augmentation strategy: {self.strategy}")
            return self
        if self.strategy in ['smote', 'ctgan', 'tvae']:
            print("Class distribution after augmentation:")
            print(self.df[self.target_column].value_counts())
        return self
