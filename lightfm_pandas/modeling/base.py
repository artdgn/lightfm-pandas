import logging
import os
import abc
import copy
import functools
import multiprocessing.pool

import numpy as np
import pandas as pd
from scipy import stats

from lightfm_pandas.utils import instrumentation, array_math
from lightfm_pandas.data import interactions
from lightfm_pandas import evaluation

logger = logging.getLogger(__name__)


class BaseDFRecommender(abc.ABC, instrumentation.LogLongCallsMeta):
    default_model_params = {}
    default_fit_params = {}

    def __init__(self, user_col='userid', item_col='itemid',
                 rating_col='rating', prediction_col='prediction',
                 model_params=None, fit_params=None, **kwargs):
        self._user_col = user_col
        self._item_col = item_col
        self._item_col_simil = item_col + '_source'
        self._rating_col = rating_col
        self._prediction_col = prediction_col
        self.model_params = self.default_model_params.copy()
        self.fit_params = self.default_fit_params.copy()
        self._set_model_params(model_params)
        self._set_model_params(kwargs)
        self._set_fit_params(fit_params)
        # self.train_df = None
        self.model = None

    @staticmethod
    def toggle_mkl_blas_1_thread(state):
        if state:
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
        else:
            os.environ.pop('MKL_NUM_THREADS', None)
            os.environ.pop('OPENBLAS_NUM_THREADS', None)

    @staticmethod
    def _dict_update(d, u):
        d = d.copy()
        if u:
            d.update(u)
        return d

    @staticmethod
    def _pop_set_dict(d, params, pop_params):
        for p in pop_params:
            if p in params:
                d[p] = params.pop(p)
        return params

    def _pop_set_params(self, params, pop_params):
        return self._pop_set_dict(self.__dict__, params, pop_params)

    def _set_model_params(self, params):
        self.model_params = self._dict_update(self.model_params, params)

    def _set_fit_params(self, params):
        self.fit_params = self._dict_update(self.fit_params, params)

    def set_params(self, **params):
        """
        this is for skopt / sklearn compatibility
        """
        # pop-set fit_params if provided in bulk
        self._set_fit_params(params.pop('fit_params', {}))
        # pop-set model_params if provided in bulk
        self._set_model_params(params.pop('model_params', {}))
        # pop-set fit_params by keys from default_fit_params if provided flat
        params = self._pop_set_dict(self.fit_params, params, self.default_fit_params.keys())
        # the rest are assumed to be model_params provided flat
        self._set_model_params(params)

    @abc.abstractmethod
    def fit(self, train_obs: interactions.ObservationsDF, *args, **kwargs):
        return self

    @abc.abstractmethod
    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10,
            exclusions=True,
            results_format='lists',
            **kwargs):
        return pd.DataFrame()

    @abc.abstractmethod
    def eval_on_test_by_ranking(
            self, test_dfs, test_names=('',), prefix='rec ',
            include_train=True, items_filter=None, k=10,
            **kwargs):
        return pd.DataFrame()

    @staticmethod
    def _flat_df_to_lists(df, sort_col, group_col, n_cutoff, target_columns):
        order = [group_col] + target_columns
        return df[order]. \
            sort_values(sort_col, ascending=False). \
            groupby(group_col). \
            aggregate(lambda x: list(x)[:n_cutoff]). \
            reset_index()

    def _recos_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._user_col,
            target_columns=[self._item_col, self._prediction_col])

    def _simil_flat_to_lists(self, df, n_cutoff=None):
        return self._flat_df_to_lists(
            df,
            n_cutoff=n_cutoff,
            sort_col=self._prediction_col,
            group_col=self._item_col_simil,
            target_columns=[self._item_col, self._prediction_col])

    def _format_results_df(self, source_vec, results_format,
                           target_ids_mat=None, scores_mat=None,
                           target_ids_lists=None, scores_lists=None):

        matrices = target_ids_mat is not None and scores_mat is not None
        lists = target_ids_lists is not None and scores_lists is not None
        assert matrices or lists, 'Provide either matrices or lists'

        if 'recommendations' in results_format:
            source_col = self._user_col
            target_col = self._item_col
            scores_col = self._prediction_col
        elif 'similarities' in results_format:
            source_col = self._item_col_simil
            target_col = self._item_col
            scores_col = self._prediction_col
        else:
            raise NotImplementedError('results_format: ' + results_format)

        order = [source_col, target_col, scores_col]

        if 'lists' in results_format:
            if matrices:
                return pd.DataFrame({
                    source_col: source_vec,
                    target_col: list(target_ids_mat),
                    scores_col: list(scores_mat)
                })[order]
            else:
                return pd.DataFrame({
                    source_col: source_vec,
                    target_col: target_ids_lists,
                    scores_col: scores_lists
                })[order]

        elif 'flat' in results_format:
            if matrices:
                n_rec = target_ids_mat.shape[1]
                return pd.DataFrame({
                    source_col: np.array(source_vec).repeat(n_rec),
                    target_col: np.concatenate(list(target_ids_mat) if len(target_ids_mat) else [[]]),
                    scores_col: np.concatenate(list(scores_mat) if len(target_ids_mat) else [[]]),
                })[order]
            else:
                return pd.DataFrame({
                    source_col: np.array(source_vec).repeat(list(map(len, target_ids_lists))),
                    target_col: np.concatenate(target_ids_lists if len(target_ids_lists) else [[]]),
                    scores_col: np.concatenate(scores_lists if len(scores_lists) else [[]]),
                })[order]

        else:
            raise NotImplementedError('results_format: ' + results_format)


class BaseDFSparseRecommender(BaseDFRecommender):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sparse_mat_builder = None
        self.train_mat = None
        self.exclude_mat = None
        self.external_features_mat = None
        self.items_handler = None

    def user_inds(self, user_ids):
        return self.sparse_mat_builder.uid_encoder.transform(user_ids)

    def user_ids(self, user_inds):
        return self.sparse_mat_builder.uid_encoder.inverse_transform(user_inds)

    def item_inds(self, item_ids):
        return self.sparse_mat_builder.iid_encoder.transform(item_ids)

    def item_ids(self, item_inds):
        return self.sparse_mat_builder.iid_encoder.inverse_transform(item_inds)

    def unknown_users_mask(self, user_ids):
        return self.sparse_mat_builder.uid_encoder.find_new_labels(user_ids)

    def unknown_items_mask(self, item_ids):
        return self.sparse_mat_builder.iid_encoder.find_new_labels(item_ids)

    def _set_data(self, train_obs, exclude_obs=None, exclude_training=True):
        """
        :param train_obs: training observations handler
        :param exclude_obs: additional observations that need to be excluded
            (e.g. observations that were not in training, but should not be predicted for test)
        :param exclude_training: whether to include training in the
            resulting excluded matrix, if True exclude_obs and train_obs are excluded
        """
        train_df = train_obs.df_obs
        self.sparse_mat_builder = train_obs.get_sparse_matrix_helper()
        self.all_users = train_df[self.sparse_mat_builder.uid_source_col].unique().astype(str)
        self.all_items = train_df[self.sparse_mat_builder.iid_source_col].unique().astype(str)
        # shuffling because np.unique() returns elements in almost sorted order by counts,
        # and it's probably not a good thing: it changes regional sparsity,
        # and at a later stage might be sampled / iterated in order
        np.random.shuffle(self.all_users)
        np.random.shuffle(self.all_items)
        self.train_mat = self.sparse_mat_builder.\
            build_sparse_interaction_matrix(train_df)
        # set training mat as default exclude_mat
        self.set_exclude_mat(exclude_obs, exclude_training=exclude_training)
        self.set_items_handler(train_obs)
        
    def set_items_handler(self, train_obs):
        if isinstance(train_obs,
                      interactions.ItemsFeaturesHandler):
            self.items_handler = interactions.ItemsFeaturesHandler(
                df_items=train_obs.df_items, item_id_col=train_obs.item_id_col)

    def set_exclude_mat(self, exclude_obs=None, exclude_training=True):
        """
        this methods allows to alter the exclusion matrix from the default of using just the training.
        An error is raised if parameters dictate no excluded data in order to prevent accidental mistake.
        :param exclude_obs: alternative / additional excluded data
        :param exclude_training: whether to exclude training data
        """
        if exclude_obs is None:
            if not exclude_training:
                raise ValueError("Got exclude_training=False, but no "
                                 "replacement exclude_obs provided. "
                                 "If you're sure you don't want to have excluded data "
                                 "at all provide an empty observation handler")
            exclude_mat = self.train_mat.copy() if self.train_mat is not None else None
        else:
            exclude_mat = self.sparse_mat_builder.\
                build_sparse_interaction_matrix(exclude_obs.df_obs)
            if exclude_training and self.train_mat is not None:
                exclude_mat += self.train_mat
        self.exclude_mat = exclude_mat

    def _reuse_data(self, other):
        self.all_users = other.all_users
        self.all_items = other.all_items
        self.sparse_mat_builder = other.sparse_mat_builder
        self.train_mat = other.train_mat
        self.exclude_mat = other.exclude_mat
        self.items_handler = other.items_handler

    def remove_unseen_users(self, user_ids, message_prefix=''):
        return self._filter_array(
            user_ids,
            encoder=self.sparse_mat_builder.uid_encoder,
            message_prefix=message_prefix,
            message_suffix='users that were not in training set.')

    def remove_unseen_items(self, item_ids, message_prefix=''):
        return self._filter_array(
            item_ids,
            encoder=self.sparse_mat_builder.iid_encoder,
            message_prefix=message_prefix,
            message_suffix='items that were not in training set.')

    @staticmethod
    def _filter_array(array, encoder, message_prefix='', message_suffix=''):
        array = np.array(array).astype(str)
        new_labels_mask = encoder.find_new_labels(array)
        n_discard = np.sum(new_labels_mask)
        if n_discard > 0:
            logger.info(
                '%s Discarding %d (out of %d) %s' %
                (message_prefix, int(n_discard), len(array), message_suffix))
        return array[~new_labels_mask]

    @staticmethod
    def _remove_self_similarities(flat_df, col1, col2):
        return flat_df[flat_df[col1].values != flat_df[col2].values].copy()

    @staticmethod
    def _eval_on_test_by_ranking_LFM(train_ranks_func, test_ranks_func,
                                     test_dfs, test_names=None, prefix='',
                                     include_train=True, k=10):
        """
        this is just to avoid the same flow twice (or more)
        :param train_ranks_func: function that return the ranks and sparse mat of training set
        :param test_ranks_func: function that return the ranks and sparse mat of a test set
        :param test_dfs: test dataframes
        :param test_names: test dataframes names
        :param prefix: prefix for this report
        :param include_train: whether to evaluate training or not
        :return: a report dataframe
        """

        # test
        if isinstance(test_dfs, pd.DataFrame):
            test_dfs = [test_dfs]

        if test_names is None or len(test_names) != len(test_dfs):
            logger.warning('No test names provided or number of names '
                           'not matching number of test DFs, using empty strings')
            test_names = [''] * len(test_dfs)

        res = []
        report_dfs = []
        full_reports = {}
        with multiprocessing.pool.ThreadPool(len(test_dfs) + int(include_train)) as pool:
            if include_train:
                res.append((prefix + 'train', pool.apply_async(train_ranks_func)))

            for test_df, test_name in zip(test_dfs, test_names):
                res.append((prefix + test_name + 'test',
                            pool.apply_async(test_ranks_func, args=(test_df,))))

            for name, r in res:
                ranks_mat, sp_mat = r.get()
                means_report, full_report = evaluation.mean_scores_report_on_ranks(
                    ranks_list=[ranks_mat], datasets=[sp_mat],
                    dataset_names=[name], k=k)
                report_dfs.append(means_report)
                full_reports.update(full_report)

        report_df = pd.concat(report_dfs, sort=False)
        return report_df, full_reports

    def get_prediction_mat_builder_adapter(
            self, mat_builder: interactions.InteractionMatrixBuilder):
        mat_builder = copy.deepcopy(mat_builder)
        mat_builder.uid_source_col = self._user_col
        mat_builder.iid_source_col = self._item_col
        mat_builder.rating_source_col = self._prediction_col
        return mat_builder

    @abc.abstractmethod
    def _get_recommendations_flat(self, user_ids, n_rec, item_ids=None,
                                  exclusions=True, **kwargs):
        return pd.DataFrame()

    @abc.abstractmethod
    def _predict_on_inds_dense(self, user_inds, item_inds):
        return np.array([])

    def get_recommendations(
            self, user_ids=None, item_ids=None, n_rec=10,
            exclusions=True, results_format='lists',
            **kwargs):

        if user_ids is not None:
            user_ids = self.remove_unseen_users(
                user_ids, message_prefix='get_recommendations: ')
        else:
            user_ids = self.all_users

        if item_ids is not None:
            item_ids = self.remove_unseen_items(
                item_ids, message_prefix='get_recommendations: ')

        recos_flat = self._get_recommendations_flat(
            user_ids=user_ids, item_ids=item_ids, n_rec=n_rec,
            exclusions=exclusions)

        if results_format == 'flat':
            return recos_flat
        else:
            return self._recos_flat_to_lists(recos_flat, n_cutoff=n_rec)

    def _check_item_ids_args(self, item_ids, target_item_ids):
        item_ids = self.remove_unseen_items(item_ids) \
            if item_ids is not None else self.all_items
        target_item_ids = self.remove_unseen_items(target_item_ids) \
            if target_item_ids is not None else None
        return item_ids, target_item_ids

    def _check_user_ids_args(self, user_ids, target_user_ids):
        user_ids = self.remove_unseen_users(user_ids) \
            if user_ids is not None else self.all_users
        target_user_ids = self.remove_unseen_users(target_user_ids) \
            if target_user_ids is not None else None
        return user_ids, target_user_ids

    def eval_on_test_by_ranking(self, test_dfs, test_names=None, prefix='rec ',
                                include_train=True, items_filter=None,
                                n_rec=100, k=10, return_full_metrics=False):

        # convert to list
        if isinstance(test_dfs, (pd.DataFrame, interactions.ObservationsDF)):
            test_dfs = [test_dfs]

        # convert to dfs
        for i in range(len(test_dfs)):
            if isinstance(test_dfs[i], interactions.ObservationsDF):
                test_dfs[i] = test_dfs[i].df_obs

        mat_builder = self.sparse_mat_builder
        pred_mat_builder = self.get_prediction_mat_builder_adapter(mat_builder)

        all_test_users = np.unique(np.concatenate([df[self._user_col].unique()
                                                   for df in test_dfs]))
        users = self.remove_unseen_users(all_test_users)

        if include_train:
            recos_flat_train = self.get_recommendations(
                user_ids=users,
                item_ids=None,
                n_rec=min(n_rec, mat_builder.n_cols),
                exclusions=False,
                results_format='flat')

        recos_flat_test = self.get_recommendations(
            user_ids=users,
            item_ids=items_filter,
            n_rec=min(n_rec, mat_builder.n_cols),
            exclusions=True,
            results_format='flat')

        ranks_all_test = pred_mat_builder.predictions_df_to_sparse_ranks(
            recos_flat_test)

        def _get_training_ranks():
            users_inds = mat_builder.uid_encoder.transform(users)
            users_inds.sort()
            # train_df = self.train_df[self.train_df[self._user_col].isin(users)].copy()
            # sp_train = self.sparse_mat_builder. \
            #     build_sparse_interaction_matrix(train_df).tocsr()
            sp_train = mat_builder.crop_rows(self.train_mat, inds_stay=users_inds)
            sp_train_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(
                sp_train, pred_mat_builder.predictions_df_to_sparse_ranks(recos_flat_train))
            return sp_train_ranks, sp_train

        def _get_test_ranks(test_df):
            sp_test = self.sparse_mat_builder. \
                build_sparse_interaction_matrix(test_df).tocsr()
            sp_test_ranks = pred_mat_builder. \
                filter_all_ranks_by_sparse_selection(sp_test, ranks_all_test)
            return sp_test_ranks, sp_test

        report_df, full_metrics = self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_ranks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train,
            k=k)

        if return_full_metrics:
            return report_df, full_metrics
        else:
            return report_df

    def get_recommendations_exact(
            self, user_ids, item_ids=None, n_rec=10,
            exclusions=True, chunksize=200, results_format='lists'):

        calc_func = functools.partial(
            self._get_recommendations_exact,
            n_rec=n_rec,
            item_ids=item_ids,
            exclusions=exclusions,
            results_format=results_format)

        chunksize = int(35000 * chunksize / self.sparse_mat_builder.n_cols)

        ret = array_math.map_batches_multiproc(calc_func, user_ids, chunksize=chunksize)
        return pd.concat(ret, axis=0, sort=False)

    def _predict_for_users_dense(self, user_ids, item_ids=None, exclusions=True):
        """
        method for calculating prediction for a grid of users and items.
        this method uses the _predict_on_inds() method that the subclasses should implement.

        :param user_ids: users ids
        :param item_ids: item ids, when None - all known items are used
        :param exclusions: when True uses default exclusion matrix sets prediction on training examples to -np.inf
        :return: a matrix of predictions (n_users, n_items)
        """

        if item_ids is None:
            item_ids = self.sparse_mat_builder.iid_encoder.classes_

        user_inds = self.user_inds(user_ids)
        item_inds = self.item_inds(item_ids)

        full_pred_mat = self._predict_on_inds_dense(user_inds, item_inds)

        if exclusions:
            exclude_mat_sp_coo = self.exclude_mat[user_inds, :][:, item_inds].tocoo()
            full_pred_mat[exclude_mat_sp_coo.row, exclude_mat_sp_coo.col] = -np.inf

        return full_pred_mat

    _predict_for_users_dense_direct = _predict_for_users_dense

    def _get_recommendations_exact(self, user_ids, item_ids=None, n_rec=10, exclusions=True,
                                   results_format='lists'):

        full_pred_mat = self._predict_for_users_dense(user_ids, item_ids,
                                                      exclusions=exclusions)

        top_inds, top_scores = array_math.top_N_sorted(full_pred_mat, n=n_rec)

        # best_ids = self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)
        best_ids = item_ids[top_inds] if item_ids is not None else \
            self.sparse_mat_builder.iid_encoder.inverse_transform(top_inds)

        return self._format_results_df(
            source_vec=user_ids, target_ids_mat=best_ids,
            scores_mat=top_scores, results_format='recommendations_' + results_format)


    def predict_for_user(self, user_id, item_ids, rank_training_last=True,
                         sort=True, combine_original_order=False):
        """
        method for predicting for one user for a small subset of items.
        optimized for minimal latency for use in real-time ranking
        will return -np.inf for combinations of unknown user / unknown items

        :param user_id: a single user ID, may be an unknown users (all predictions will be None)
        :param item_ids: a subset of item IDs, may have unknown items (prediction for those will be None)
        :param rank_training_last:  if set to True predictions for interactions seen during training
            seen during will be ranked last by being set to -np.inf
        :param combine_original_order: whether to combine predictions with original
            order of the items (if they were already ordered in a meaningful way)
        :param sort: whether to sort the result in decreasing order of prediction (best first)
        :return: a pandas DataFrame of userid | itemid | prediction,
            sorted in decresing order of prediction (if sort=True),
            with Nones for unknown users or items
        """

        user_id = str(user_id)
        item_ids = np.array(item_ids).astype(str)

        df = pd.DataFrame()
        df[self._item_col] = item_ids  # assigning first because determines length
        df[self._user_col] = user_id
        df[self._prediction_col] = -np.inf

        if np.any(self.unknown_users_mask([user_id])):
            return df

        new_items_mask = self.unknown_items_mask(item_ids)

        preds = self._predict_for_users_dense_direct(
            user_ids=[user_id],
            item_ids=item_ids[~new_items_mask],
            exclusions=rank_training_last)

        df[self._prediction_col].values[~new_items_mask] = preds.ravel()

        if combine_original_order:
            orig_score = len(item_ids) - np.arange(len(item_ids))
            preds_score = df[self._prediction_col].values
            df[self._prediction_col] = (stats.rankdata(orig_score) +
                                        stats.rankdata(preds_score)) / 2

        if sort:
            df.sort_values(self._prediction_col, ascending=False, inplace=True)

        return df[[self._user_col, self._item_col, self._prediction_col ]]  # reordering


class BasePredictorRecommender(BaseDFSparseRecommender):

    @abc.abstractmethod
    def _predict_on_inds(self, user_inds, item_inds):
        pass

    @abc.abstractmethod
    def _predict_rank(self, test_mat, train_mat=None):
        pass

    def _predict_on_inds_dense(self, user_inds, item_inds):
        n_users = len(user_inds)
        n_items = len(item_inds)
        user_inds_mat = user_inds.repeat(n_items)
        item_inds_mat = np.tile(item_inds, n_users)
        return self._predict_on_inds(user_inds_mat, item_inds_mat).reshape((n_users, n_items))

    def predict_on_df(self, df, exclusions=True, user_col=None, item_col=None):
        if user_col is not None and user_col != self.sparse_mat_builder.uid_source_col:
            df[self.sparse_mat_builder.uid_source_col] = df[user_col]
        if item_col is not None and item_col != self.sparse_mat_builder.iid_source_col:
            df[self.sparse_mat_builder.iid_source_col] = df[item_col]

        mat_builder = self.sparse_mat_builder
        df = mat_builder.remove_unseen_labels(df)
        df = mat_builder.add_encoded_cols(df)
        df[self._prediction_col] = self._predict_on_inds(
            df[mat_builder.uid_col].values, df[mat_builder.iid_col].values)

        if exclusions:
            exclude_mat_sp_coo = \
                self.exclude_mat[df[mat_builder.uid_col].values, :] \
                    [:, df[mat_builder.iid_col].values].tocoo()
            df[df[mat_builder.uid_col].isin(exclude_mat_sp_coo.row) &
               df[mat_builder.iid_col].isin(exclude_mat_sp_coo.col)][self._prediction_col] = -np.inf

        df.drop([mat_builder.uid_col, mat_builder.iid_col], axis=1, inplace=True)
        return df

    def eval_on_test_by_ranking_exact(self, test_dfs, test_names=None,
                                      prefix='lfm ', include_train=True, k=10,
                                      return_full_metrics=False):

        # convert to list
        if isinstance(test_dfs, (pd.DataFrame, interactions.ObservationsDF)):
            test_dfs = [test_dfs]

        # convert to dfs
        for i in range(len(test_dfs)):
            if isinstance(test_dfs[i], interactions.ObservationsDF):
                test_dfs[i] = test_dfs[i].df_obs

        def _get_training_ranks():
            all_test_users = np.unique(np.concatenate(
                [df[self._user_col].unique() for df in test_dfs]))
            users = self.remove_unseen_users(all_test_users)
            users_inds = self.sparse_mat_builder.uid_encoder.transform(users)
            users_inds.sort()
            sub_train_mat = self.sparse_mat_builder.crop_rows(
                self.train_mat, inds_stay=users_inds)
            ranks_mat = self._predict_rank(sub_train_mat)
            return ranks_mat, sub_train_mat

        def _get_test_ranks(test_df):
            test_sparse = self.sparse_mat_builder.\
                build_sparse_interaction_matrix(test_df)
            ranks_mat = self._predict_rank(test_sparse, self.train_mat)
            return ranks_mat, test_sparse

        report_df, full_metrics = self._eval_on_test_by_ranking_LFM(
            train_ranks_func=_get_training_ranks,
            test_ranks_func=_get_test_ranks,
            test_dfs=test_dfs,
            test_names=test_names,
            prefix=prefix,
            include_train=include_train,
            k=k)

        if return_full_metrics:
            return report_df, full_metrics
        else:
            return report_df
