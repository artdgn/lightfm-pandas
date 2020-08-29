"""
using multiple test sets
"""

# dataset: download and prepare dataframes
import pandas as pd
from lightfm_pandas.data.datasets.prep_movielense_data import get_and_prep_data
from lightfm_pandas.modeling.lightfm import LightFMRecommender

rating_csv_path, users_csv_path, movies_csv_path = get_and_prep_data()

# read the interactions dataframe and create a data handler object and  split to train and test
ratings_df = pd.read_csv(rating_csv_path)
from lightfm_pandas.data.interactions import ObservationsDF

obs = ObservationsDF(ratings_df)
train_obs, test_obs = obs.split_train_test(ratio=0.2, users_ratio=0.2)

def construct_multiple_test_sets(test_df, train_df):
    # by user history - active and inactive users
    user_hist_counts = train_df.userid.value_counts()
    user_hist_counts.hist(bins=100, alpha=0.5)
    active_users = user_hist_counts[user_hist_counts >= 300].index.tolist()
    test_df_act_us = test_df[test_df.userid.isin(active_users)]
    test_df_nonact_us = test_df[~test_df.userid.isin(active_users)]

    # by item popularity- popular and unpopular items
    item_hist_counts = train_df.itemid.value_counts()
    item_hist_counts.hist(bins=100, alpha=0.5)
    popular_items = item_hist_counts[item_hist_counts >= 1000].index.tolist()
    test_df_pop_movies = test_df[test_df.itemid.isin(popular_items)]
    test_df_nonpop_movies = test_df[~test_df.itemid.isin(popular_items)]

    test_dfs = [test_df, test_df_act_us, test_df_nonact_us, test_df_pop_movies, test_df_nonpop_movies]
    test_names = ['all ', 'active users ', 'inactive users ', 'popular movies ', 'unpopular movies ']
    df_lens = [len(t) for t in test_dfs]
    print('Test DFs counts: ' + str(list(zip(test_names, df_lens))))
    return test_dfs, test_names

test_dfs, test_names = construct_multiple_test_sets(test_df=test_obs.df_obs, train_df=train_obs.df_obs)

# evaluation
lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)
print(lfm_rec.eval_on_test_by_ranking(
    test_dfs=test_dfs, test_names=test_names, prefix='lfm regular '))



"""
Example explaining the peculiriaties of evaluation
"""

from lightfm_pandas.data.datasets.prep_movielense_data import get_and_prep_data
import pandas as pd
from lightfm_pandas.data.interactions import ObservationsDF
from lightfm_pandas.modeling.lightfm import LightFMRecommender

rating_csv_path, _, _ = get_and_prep_data()
ratings_df = pd.read_csv(rating_csv_path)

obs = ObservationsDF(ratings_df, uid_col='userid', iid_col='itemid')
train_obs, test_obs = obs.split_train_test(ratio=0.2)

# train and test LightFM recommender
lfm_rec = LightFMRecommender()
lfm_rec.fit(train_obs, epochs=10)

# print evaluation results:
# for LightFM there is an exact method that on large and sparse
# data might be too slow (for this data it's much faster though)
print(lfm_rec.eval_on_test_by_ranking_exact(test_obs.df_obs, prefix='lfm regular exact '))

# this ranking evaluation is done by sampling top n_rec recommendations
# rather than all ranks for all items (very slow and memory-wise expensive for large data).
# choosing higher values for n_rec makes
# the evaluation more accurate (less pessimmistic)
# this way the evaluation is mostly accurate for the top results,
# and is quite pessimmistic (especially for AUC, which scores for all ranks) and any non @k metric
print(lfm_rec.eval_on_test_by_ranking(test_obs.df_obs, prefix='lfm regular ', n_rec=100))
