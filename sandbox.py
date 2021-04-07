# nopep8 for the lines below, needs to be executed first
import os  # nopep8

from IPython import get_ipython  # nopep8

get_ipython().run_line_magic("load_ext", "autoreload")  # nopep8
get_ipython().run_line_magic("autoreload", "2")  # nopep8
# os.chdir("/projects/customer_checkin")  # nopep8

import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 10)
