def predict_chems(path_to_model, predction_folder_path, chemicals, model_versions, path_to_spectra, data_version,path_to_splits=None):
    data_spc = pd.read_csv(os.path.join(path_to_spectra, "spc/fritsch.csv"), index_col=0, engine='c')
 
   
    for model_version in model_versions:
        if (model_version not in [str(x) for x in os.listdir(path_to_model)]):
            continue

        print(f'Starting prediction using model version {model_version}')
        base_path = Path(path_to_model)
        print(f"This is the path to model {path_to_model}")
        for chemical in chemicals:
            data = data_spc.copy(deep=True)
            print("Making predictions for these number of spectra: ",len(data))
            preds_comb = pd.DataFrame()
            models_folder = base_path / model_version / chemical / 'std'
            print(f"This is the model path {models_folder}")
            all_models = [x for x in models_folder.glob('**/*.hdf5')]
            print(f"These are the models {all_models}")
            if len(all_models) == 0:
                raise("Models for chemical {} not found".format(chemical))
            #                             data = pd.read_csv(filename, index_col=[0,1])
            data = data
            new_indices = data.index
            # data.drop(chemical, axis=1, inplace=True)

            for model_path in all_models:

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
                        print("------------------------------.",key_)
                        if key_ == "EPO":
                            train = train.sub(train.mean(axis=1), axis=0).div(train.std(axis=1), axis=0)
                            epo_matrix = pd.read_csv(os.path.join(path_to_model, model_version, "epo_matrix.csv"),index_col=0)
                            train = np.dot(train.values, epo_matrix.values)
                            continue
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
            preds_comb = preds_comb.dropna()
            os.makedirs(
                f'{predction_folder_path}', exist_ok=True)
            preds_comb.to_csv(
                f'{predction_folder_path}/{chemical}_{data_version}_{model_version}_preds.csv')
        print(f'Finalizing prediction using model version {model_version} for data version {data_version} ')



### SNV
import pandas
import numpy
df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
epo_matrix = pd.read_csv("path_to_model_v1.1/epo_matrix.csv",index_col=0)
df = np.dot(df.values, epo_matrix.values)
