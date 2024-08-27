from sklearn.metrics import *
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow import keras
import os
import csv
import json
# import pyodbc
import joblib
from pandas import pivot_table
from distutils.dir_util import copy_tree
import shutil
import itertools
import pandas as pd
from datetime import date
from PIL import Image
import glob
import numpy as np
from imageio import imsave
from math import log10, floor, ceil
from scipy.stats import linregress
from sklearn.metrics import accuracy_score, f1_score, median_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.figure_factory as ff
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import tensorflow as tf
# from wai.ma.core.matrix import Matrix, helper
# from wai.ma.transformation import SavitzkyGolay2
# from wai.ma.filter import Downsample
from pathlib import Path
import utils
print(f'tensorflow version : {tf.version.VERSION}')
# tf.enable_eager_execution()

# ClassifierKwargs = ClassifierKwargs()


def predict_chems(path_to_model, predction_folder_path, chemicals, model_versions, data):
    for model_version in model_versions:

        print(f'Starting prediction using model version {model_version}')
        base_path = Path(path_to_model)
        for chemical in chemicals:
            chems = [
                'aluminium',
                'phosphorus',
                'ph',
                'exchangeable_acidity',
                'calcium',
                'magnesium',
                'sulphur',
                'sodium',
                'iron',
                'manganese',
                'boron',
                'copper',
                'zinc',
                'total_nitrogen',
                'potassium',
                'ec_salts',
                'organic_carbon', 'cec',
                'sand', 'silt', 'clay'
            ]
            if chemical not in chems:
                continue
            print(chemical)
            preds_comb = pd.DataFrame()
            models_folder = base_path / chemical / 'std'
            all_models = [x for x in models_folder.glob('**/*.hdf5')]

            data = data
            new_indices = data.index

            for model_path in all_models:
                print(model_path)
                json_path = model_path.parent.parent / 'model.json'

                with open(json_path) as f:
                    json_ = json.load(f)

                inputs = []

                for i in range(len(json_['Inputs'])):
                    input_name = json_['Inputs'][i]['Name']
#                     print(f'filename: {filename}')
                    train = data.copy(deep=True)
                    for j in range(len(json_['Inputs'][i]['Pre-processing'])):
                        key_ = json_['Inputs'][i]['Pre-processing'][j]['Name']
                        if input_name == 'nir2':
                            input_name = 'nir.2'

                        pickle_path = model_path.parent / 'preprocess' / \
                            f'input.{input_name}.{j}.{key_}.pickle'
                        pickle_ = joblib.load(pickle_path)
                        train = pickle_.fit_transform(train)

                    inputs.append(train.values)

                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(model_path, compile=False)
                preds = pd.DataFrame(model(inputs).numpy())
                preds_comb = pd.concat([preds_comb, preds], axis=1)
            preds_comb = preds_comb.median(axis=1)

            preds_comb.index = new_indices

            os.makedirs(
                f'{predction_folder_path}/{model_version}', exist_ok=True)
            preds_comb.to_csv(
                f'{predction_folder_path}/{model_version}/{chemical}_preds.csv')
        print(f'Finalizing prediction using model version {model_version}')


# predict_chems('dl_models_all_chems_20210414/dl_v2.2_update_2022',
#               '../DSML87/outputFiles/preds', ['total_nitrogen', 'clay'], ['DLv2.2'], pd.read_csv('../outputFiles/spectraldata.csv'))
