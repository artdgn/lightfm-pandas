import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import sparse as sp
import sklearn_pandas

from lightfm_pandas.utils import instrumentation


class ExternalFeaturesDF(instrumentation.LogLongCallsMeta):
    """
    handles external items features and feature engineering
    """

    _numeric_duplicate_suffix = '_num'
    _item_ind_col = '_item_ind'

    def __init__(self, feat_df, id_col, num_cols=None, cat_cols=None, bin_cols=None):
        self.feat_df = feat_df.copy()
        self.id_col = id_col
        self.num_cols = num_cols if num_cols is not None else []
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.bin_cols = bin_cols if bin_cols is not None else []
        self._numeric_duplicate_cols = None
        self._feat_weight = None
        self.df_transformer = None
        if not self.num_cols and not self.cat_cols and not self.bin_cols:
            self.infer_columns_types()

    def infer_columns_types(self,
                            categorical_unique_ratio=0.05,
                            categorical_n_unique=20):

        len_df = len(self.feat_df)

        if not len_df:
            raise ValueError('Features DF is empty')

        feat_cols = self.feat_df.columns.difference([self.id_col])

        self.num_cols, self.cat_cols, self.bin_cols = [], [], []

        for col in feat_cols:
            if str(self.feat_df[col].dtype) in ['O', 'o']:
                self.cat_cols.append(col)
            else:
                n_unique = self.feat_df[col].nunique()
                if n_unique == 1:
                    continue  # fixed value column
                if n_unique == 2:
                    self.bin_cols.append(col)  # binary column
                else:
                    unique_ratio = n_unique / len_df
                    if n_unique < categorical_n_unique or \
                            unique_ratio <= categorical_unique_ratio:
                        self.cat_cols.append(col)
                    else:
                        self.num_cols.append(col)

    def apply_selection_filter(self, selection_filter=None):
        if selection_filter is not None and len(selection_filter) >= 1:
            # no selection applied for None, '', []
            self.cat_cols = [col for col in self.cat_cols if col in selection_filter]
            self.num_cols = [col for col in self.num_cols if col in selection_filter]
            self.bin_cols = [col for col in self.bin_cols if col in selection_filter]

        self.feat_df = self.feat_df[[self.id_col] + self.cat_cols + self.num_cols + self.bin_cols]

        return self

    def _check_intersecting_column_names(self):
        self._numeric_duplicate_cols = list(set(self.cat_cols).intersection(set(self.num_cols)))
        if len(self._numeric_duplicate_cols):
            for col in self._numeric_duplicate_cols:
                alt_name = col + self._numeric_duplicate_suffix
                self.feat_df[alt_name] = self.feat_df[col].copy()
                self.num_cols.remove(col)
                self.num_cols.append(alt_name)

    def create_sparse_features_mat(self,
                                   items_encoder,
                                   mode='binarize',
                                   normalize_output=False,
                                   add_identity_mat=False,
                                   numeric_n_bins=128,
                                   feat_weight=1.0):
        """
        creates a sparse feature matrix from item features

        :param items_encoder: the encoder that is used to filter and
            align the features dataframe to the sparse matrices
        :param mode: 'binarize' or 'encode'.
            'binarize' (default) - creates a binary matrix by label binarizing
            categorical and numeric feature.
            'encode' - only encodes the categorical features to integers and leaves numeric as is
        :param add_identity_mat: indicator whether to add a sparse identity matrix
            (as used when no features are used), as per LightFM's docs suggestion
        :param normalize_output:
            None (default) - no normalization
            'rows' - normalize rows with l1 norm
            anything else - normalize cols with l1 norm
        :param numeric_n_bins: number of bins for binning numeric features
        :param feat_weight:
            feature weight relative to identity matrix (can be used to emphasize one or the other)
            can also be a dictionary of weights to be applied to columns e.g. {'column_name': 10}

        :return: sparse feature mat n_items x n_features
        """

        self._check_intersecting_column_names()

        feat_df = self.feat_df[
            [self.id_col] + self.cat_cols + self.num_cols + self.bin_cols]
        # get only features for relevant items
        feat_df = feat_df[feat_df[self.id_col].isin(items_encoder.classes_)]
        # convert from id to index
        feat_df[self._item_ind_col] = items_encoder.transform(feat_df[self.id_col])

        # reorder in correct index order
        n_items = len(items_encoder.classes_)
        full_feat_df = pd.merge(
            pd.DataFrame({self._item_ind_col: np.arange(n_items)}),
            feat_df.drop([self.id_col], axis=1), on=self._item_ind_col, how='left'). \
            drop_duplicates(self._item_ind_col). \
            set_index(self._item_ind_col, drop=True)

        # remove nans resulting form join
        # https://stackoverflow.com/questions/34913590/fillna-in-multiple-columns-in-place-in-python-pandas
        full_feat_df = full_feat_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

        full_feat_df[self.cat_cols] = \
            full_feat_df[self.cat_cols].astype(str)

        if len(full_feat_df):
            self.df_transformer = self.init_df_transformer(
                mode=mode,
                categorical_feat_cols=self.cat_cols,
                numeric_feat_cols=self.num_cols,
                binary_feat_cols=self.bin_cols,
                numeric_n_bins=numeric_n_bins)

            feat_mat = self.df_transformer.fit_transform(full_feat_df)

            if sp.issparse(feat_mat):
                feat_mat.eliminate_zeros()

            # weight the features before adding the identity mat
            self._feat_weight = feat_weight
            feat_mat = self._apply_weights_to_matrix(feat_mat)

            # normalize each row
            if normalize_output:
                axis = int(normalize_output == 'rows')
                feat_mat = preprocessing.normalize(feat_mat, norm='l1', axis=axis, copy=False)

            if add_identity_mat:
                # identity mat
                id_mat = sp.identity(n_items, dtype=np.float32, format='csr')

                assert sp.issparse(feat_mat), 'Trying to add identity mat to non-sparse matrix'

                full_feat_mat = self.concatenate_csc_matrices_by_columns(
                    feat_mat.tocsc(), id_mat.tocsc()).tocsr()
            else:
                full_feat_mat = feat_mat

            return full_feat_mat

        else:
            return None

    def _apply_weights_to_matrix(self, feat_mat):
        if np.isscalar(self._feat_weight):
            feat_mat = feat_mat.astype(np.float32) * self._feat_weight
        elif isinstance(self._feat_weight, dict):
            for col, weight in self._feat_weight.items():
                cols_mask = np.core.defchararray.startswith(
                    self.df_transformer.transformed_names_, col)
                feat_mat[:, cols_mask] *= weight
        else:
            raise ValueError('Uknown feature weight format.')
        return feat_mat

    @staticmethod
    def concatenate_csc_matrices_by_columns(matrix1, matrix2):
        # https://stackoverflow.com/a/33259578/6485667
        new_data = np.concatenate((matrix1.data, matrix2.data))
        new_indices = np.concatenate((matrix1.indices, matrix2.indices))
        new_ind_ptr = matrix2.indptr + len(matrix1.data)
        new_ind_ptr = new_ind_ptr[1:]
        new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

        return sp.csc_matrix((new_data, new_indices, new_ind_ptr), dtype=np.float32)

    @staticmethod
    def init_df_transformer(mode, categorical_feat_cols, numeric_feat_cols, binary_feat_cols,
                            numeric_n_bins=64):
        if mode=='binarize':
            feat_mapper = sklearn_pandas.DataFrameMapper(
                [(cat_col, preprocessing.LabelBinarizer(sparse_output=True))
                 for cat_col in categorical_feat_cols] +
                [(num_col, NumericBinningBinarizer(n_bins=numeric_n_bins, sparse_output=True))
                 for num_col in numeric_feat_cols] +
                [(bin_col, preprocessing.LabelBinarizer(sparse_output=True))
                 for bin_col in binary_feat_cols],
                sparse=True
            )
        elif mode=='encode':
            feat_mapper = preprocessing.DataFrameMapper(
                [(cat_col,
                  preprocessing.LabelEncoder())
                 for cat_col in categorical_feat_cols],
                sparse=True,
                default=None  # pass other columns as is
            )
        else:
            raise NotImplementedError('Unknown transform mode')
        return feat_mapper


class NumericBinningEncoder(preprocessing.LabelEncoder):
    """
    class for label-encoding a continuous variable by binning
    """
    def __init__(self, n_bins=50):
        super().__init__()
        self.n_bins = n_bins
        self.bins = None

    def fit(self, y):
        percentiles = list(np.linspace(0, 100, num=(self.n_bins + 1)))
        self.bins = np.percentile(y, percentiles[1:])

        if len(np.unique(self.bins)) != len(self.bins):
            self.bins = list(np.linspace(
                np.min(y) - 0.001, np.max(y) + 0.001, num=(self.n_bins + 1)))

        return self

    def transform(self, y):
        inc_bins = list(self.bins)
        inc_bins[0] = min(inc_bins[0], np.min(y))
        inc_bins[-1] = max(inc_bins[-1], np.max(y))

        y_binned = pd.cut(y, bins=inc_bins, labels=False, include_lowest=True)
        y_ind = y_binned.astype(int, copy=False)
        return y_ind


class NumericBinningBinarizer(preprocessing.LabelBinarizer):
    """
    class for one-hot encoding a continuous variable by binning
    """
    def __init__(self, n_bins=50, spillage=2, **kwargs):
        """
        :param n_bins: number of bins
        :param spillage:
            number of neighbouring bins that are also activated in
            order to preserve some "proximity" relationship, default to 2
            e.g. for spillage=1 a result vec would be [.. 0, 0, 0.25, 0.5, 1, 0.5, 0.25, 0, 0 ..]
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._spillage = spillage
        self._binner = NumericBinningEncoder(n_bins=n_bins)

    def fit(self, y):
        self._binner.fit(y)
        super().fit(range(len(self._binner.bins)))
        return self

    def transform(self, y):
        y_binned = self._binner.transform(y)
        binarized = super().transform(y_binned)
        if self._spillage:
            for i in range(1, self._spillage + 1):
                binarized += super().transform(y_binned + i) / 2**i
                binarized += super().transform(y_binned - i) / 2**i
        return binarized