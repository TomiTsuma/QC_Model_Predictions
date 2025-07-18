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
from dotenv import load_dotenv
load_dotenv()


def predict_chems(path_to_model, predction_folder_path, chemicals, model_versions, data):
    for model_version in model_versions:

        print(f'Starting prediction using model version {model_version}')
        base_path = Path(path_to_model)
        for chemical in chemicals:
            print(chemical)
            preds_comb = pd.DataFrame()
            models_folder = base_path / chemical / 'std'
            print(models_folder)
            all_models = [x for x in models_folder.glob('**/*.hdf5')]

            data = data
            # if os.getenv("SENSOR_ID") == "2" and os.getenv("SAMPLE_TYPE_ID") == "2" and os.getenv("SAMPLE_PRETREATMENT_ID") == "3" and chemical not in ['total_nitrogen','total_carbon']:
            #     data = data.apply(lambda row: (row - np.mean(row)) / np.std(row, ddof=1), axis=1)
            new_indices = data.index

            for model_path in all_models:

                json_path = model_path.parent.parent / 'model.json'

                with open(json_path) as f:
                    json_ = json.load(f)

                inputs = []

                for i in range(len(json_['Inputs'])):
                    input_name = json_['Inputs'][i]['Name']
                    train = data.copy(deep=True)
                    is_epo_present = False
                    for j in range(len(json_['Inputs'][i]['Pre-processing'])):
                        
                        key_ = json_['Inputs'][i]['Pre-processing'][j]['Name']
                        
                        if key_ == "EPO":
                            is_epo_present = True
                            train = train.sub(train.mean(axis=1), axis=0).div(train.std(axis=1), axis=0)
                            epo_matrix = pd.read_csv(os.path.join(path_to_model, model_version, "epo_matrix.csv"),index_col=0)
                            train = np.dot(train.values, epo_matrix.values)
                            
                            train = pd.DataFrame(train)
                            continue

                        if input_name == 'nir2':
                            input_name = 'nir.2'
                        if is_epo_present == False:
                            position = j
                            pickle_path = model_path.parent / 'preprocess' / \
                                f'input.{input_name}.{position}.{key_}.pickle'
                        else:
                            position = j-1

                            pickle_path = model_path.parent / 'preprocess' / \
                                f'input.{input_name}.{position}.{key_}.pickle'
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


predict(
    chemicals=["ph","sodium"],
    data=pd.read_csv(f"/home/tom/DSML125/MSSC_DVC/spc/spc.csv",index_col=0, engine="c"),
    predction_folder_path=f"/home/tom/DSML125/outputFiles/predictions",
    model_versions=["v1.0"],
    path_to_model=f"/home/tom/DSML125/QC_Model_Predictions/soil-alpha-atr-epo-fritsch-2mm/v1.0",
    output_path=f"/home/tom/DSML125/outputFiles/evaluation",
    project_name= project_name, 
    path_to_splits=f"/home/tom/DSML125/MSSC_DVC/{data_item}",

    data_version = data_item
)