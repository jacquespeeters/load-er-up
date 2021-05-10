"""Unearthed Sound The Alarm Training Template"""
import argparse
import logging
import os
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join

import pandas as pd

from ensemble_model import EnsembleModel
from preprocess import preprocess


def test_expected_optim_f1():
    my_model = EnsembleModel()
    assert my_model.expected_optim_f1(pd.Series([0.01, 0.01, 0.99])).equals(
        pd.Series([False, False, True])
    )

    assert my_model.expected_optim_f1(
        pd.Series([0.01, 0.49, 0.51, 0.77, 0.88, 0.01, 0.99])
    ).equals(pd.Series([False, True, True, True, True, False, True]))
