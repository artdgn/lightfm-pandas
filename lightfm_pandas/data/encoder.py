import importlib

import numpy as np
import pandas as pd
import pandas.core.algorithms
import pandas.core.dtypes.common
import sklearn.utils
from sklearn import preprocessing


class PDLabelEncoder(preprocessing.LabelEncoder):
    """
    adapted and sped up further from here: https://github.com/scikit-learn/scikit-learn/issues/7432
    a blazingly fast version of encoder that's using Pandas encoding
    which is using hash tables instead of sorted arrays
    """

    pd_cat_module = importlib.import_module(pd.Categorical.__module__)

    def fit(self, y):
        y = sklearn.utils.column_or_1d(y, warn=True)
        _, self.classes_ = pd.factorize(y, sort=True)
        self._cat_dtype = self.pd_cat_module.CategoricalDtype(self.classes_)
        self._dtype = self._cat_dtype.categories.dtype
        self._table = self._get_table_for_categories(self.classes_, self._cat_dtype.categories)
        return self

    @classmethod
    def _get_table_for_categories(cls, values, categories):
        # ripped out of _get_codes_for_values() in pandas Categorical module
        if not cls.pd_cat_module.is_dtype_equal(values.dtype, categories.dtype):
            values = pandas.core.dtypes.common.ensure_object(values)
            categories = pandas.core.dtypes.common.ensure_object(categories)

        hash_klass, vals = pandas.core.algorithms._get_data_algo(values)
        _, cats = pandas.core.algorithms._get_data_algo(categories)
        t = hash_klass(len(cats))
        t.map_locations(cats)
        return t

    def check_is_fitted(self):
        sklearn.utils.validation.check_is_fitted(
            self, ['classes_', '_cat_dtype', '_table', '_dtype'])

    def transform(self, y, check_labels=True):
        self.check_is_fitted()
        y = sklearn.utils.column_or_1d(y, warn=True)

        ## slower because it creates the hash table every time
        # trans_y = pd.Categorical(y, dtype=self._cat_dtype).codes.copy()
        trans_y = self.pd_cat_module.coerce_indexer_dtype(
            indexer=self._table.lookup(y.astype(self._dtype)),
            categories=self._cat_dtype.categories)

        if check_labels:
            mask_new_labels = self._new_labels_locs(trans_y)
            if np.any(mask_new_labels):
                raise ValueError("y contains new labels: %s" %
                                 str(np.unique(y[mask_new_labels])))

        return trans_y

    def inverse_transform(self, y):
        y = np.array(y)
        shape = y.shape
        return super().inverse_transform(y.ravel()).reshape(shape)

    def _new_labels_locs(self, trans_y):
        return trans_y == -1

    def find_new_labels(self, y):
        return self._new_labels_locs(self.transform(y, check_labels=False))

    def __getstate__(self):
        # custom pickling behaviour because the table object is unpickleable
        state = super().__getstate__()
        state['_table'] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._table = self._get_table_for_categories(self.classes_, self._cat_dtype.categories)
