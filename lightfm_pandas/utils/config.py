import multiprocessing.pool

import pandas as pd

from lightfm_pandas.utils import logging_config

N_CPUS = multiprocessing.cpu_count()

logging_config.config()


def console_settings():
    pd.set_option('display.max_colwidth', 300)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


console_settings()
