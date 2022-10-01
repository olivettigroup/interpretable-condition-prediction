import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
import pickle as pkl
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('/home/jupyter/CJK/TempTime')
from data_extraction_rxn_classification_all_targets import *
import shap
import json
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK
import time
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from sklearn.linear_model import ElasticNet
import torch
from torch import nn, optim
from torch.nn.modules import Module
from torch.utils.data import Dataset, DataLoader

def train(model, objective, featurization):
    mapping = {
        "calcine_temp": 0,
        "sinter_temp": 1,
        "anneal_temp": 2,
        "dry_temp": 3,
        "calcine_time": 4,
        "sinter_time": 5,
        "anneal_time": 6,
        "dry_time": 7
    }
    assert model in ["xgb", "nn", "lr", "rf"]
    assert featurization in ["mp_fraction", "pca"]
    assert objective in mapping
    
    with open('/home/jupyter/CJK/TempTime/data/solid-state_dataset_20200713.json') as f:
        ss_data= json.load(f)
    ss_reactions = ss_data["reactions"]
    ss_extracted, ss_precursor_nomenclature = extract_solidstate(ss_reactions, max_pre=5)
    
    with open('/home/jupyter/CJK/TempTime/data/ss_extracted_NO_IMPUTATION_precs_all_targets.pkl', 'rb') as f:
        papers = pkl.load(f)
        
    # only use papers with a particular temperature (or time)
    obj = mapping[objective]
    papers = [x for x in papers if not np.isnan(x['temp_time_vector'][obj])]

    # mean = np.mean([x['temp_time_vector'][obj] for x in papers])
    # std = np.std([x['temp_time_vector'][obj] for x in papers])
    # papers = [x for x in papers if (mean - 3*std) <= x['temp_time_vector'][obj] <= (mean + 3*std)]
    
    targets = [x['target'] for x in papers]
    precursors = [x['precursors'] for x in papers]
    to_predict = [x['temp_time_vector'][obj] for x in papers]
    dois = [x['DOI'] for x in papers]
    
    
    # get means
    only_ss_rxns = []
    for target, precs, objs, doi in zip(targets, precursors, to_predict, dois):
        found = False
        for result in only_ss_rxns:
            if result["target"] == target and set(result["precursors"]) == set(precs):
                result[objective].append(objs)
                result["DOIs"].append(doi)
                found = True
        if not found:
            new_result = {}
            new_result["target"] = target
            new_result["precursors"] = precs
            new_result[objective] = [objs]
            new_result["DOIs"] = [doi]
            only_ss_rxns.append(new_result)
            

    # change precursor functional groups back
    for rxn in only_ss_rxns:
        for i in range(len(rxn['precursors'])):
            rxn['precursors'][i] = ss_precursor_nomenclature[Composition(rxn['precursors'][i])]
            
    if featurization=="mp_fraction":
        df, num_pts = add_mp_fraction_feats(only_ss_rxns, objective)
        prec_magpie_feats=None
    elif featurization=="pca":
       df, prec_magpie_feats, num_pts = add_pca_feats(only_ss_rxns, objective)

            
    zero_cols = [col for col, is_zero in ((df == 0).sum() == df.shape[0]).items() if is_zero]

    omit_columns = ['targets', 'precursors', objective, 'composition_obj'] + zero_cols
    X_columns = [x for x in df.columns if x not in omit_columns]
    y_column = objective
    X_df = df[X_columns]
    X_columns = list(X_df.columns)
    X = X_df.values
    y = df[y_column].values
    print("Shape of X: {}".format(X.shape))
    print("Shape of y: {}".format(y.shape))
    
    if featurization == "mp_fraction":
        X, y, n_pts = shuffle(X, y, num_pts, random_state=42)
    else:
        X, y, n_pts, prec_magpie_feats = shuffle(X, y, num_pts, prec_magpie_feats, random_state=42)
        
    n_pts = np.array(n_pts)
    
    if model == "xgb":
        return train_xgb(X, y, n_pts, featurization, prec_magpie_feats)
    elif model == "nn":
        return train_nn(X, y, n_pts, featurization, prec_magpie_feats)
    elif model == "rf":
        return train_rf(X, y, n_pts, featurization, prec_magpie_feats)
    elif model == "lr":
        return train_lr(X, y, n_pts, featurization, prec_magpie_feats)

            
def add_mp_fraction_feats(only_ss_rxns, objective):
    
    all_elements = [Element.from_Z(i).symbol for i in range(1, 119)]
            
    # Onehot precursor functional groups
    m_anions = ['H2PO4', 'HPO4', 'HCO3', 'HSO4', 'HSO3', 'C2O4']
    d_anions = ['CO3', 'PO4', 'PO3', 'OH', 'NH4', 'NO3', 'NO2', 'SO4', 'SO3', 'CN'] #BO3, VO4, NH2
    s_anions = ['F', 'B', 'P', 'Cl', 'F', 'Br', 'S', 'N', 'O', 'C']
    ions = m_anions + d_anions + s_anions + ['Org'] + ['Ac'] + ['Elem'] + ['Other']
    
    
    ss_anion_dict = {x: [] for x in ions}
    for rxn in only_ss_rxns:
        anion_dict = {x: [] for x in ions}
        for prec in rxn['precursors']:
            found = False
            # check acetate first
            if 'CH3COO' in prec:
                anion_dict['Ac'].append(prec)
                found = True
            # check organic
            elif all(Element(x) in Composition(prec).elements for x in ['C', 'H', 'O']) and Element('N') not in Composition(prec).elements:
                anion_dict['Org'].append(prec)
                found = True
            else:
                for anion in ions:
                    # special check for elemental anions
                    if (anion in all_elements and Element(anion) in Composition(prec).elements) or (anion not in all_elements and anion in prec):
                        anion_dict[anion].append(prec)
                        found = True
                        break
            if not found:
                # check if precursor is elemental
                if prec in all_elements:
                    anion_dict['Elem'].append(prec)
                else:
                    anion_dict['Other'].append(prec)
        # create fractional embedding

        for key in anion_dict:
            ss_anion_dict[key].append(len(anion_dict[key]) / len(rxn['precursors']))
            
            

    with open('/home/jupyter/CJK/TempTime/rxn_classification_fraction/data/full_mp.json', 'r') as f:
        full_mp = json.load(f)
        
    full_mp = {Composition(x): full_mp[x] for x in full_mp}
    full_mp_median = np.median(list(full_mp.values()))
    
    # add MP features
    for new_result in only_ss_rxns:
        # precursor MP features
        melting_points = [full_mp.get(Composition(x), full_mp_median) for x in new_result["precursors"]]
        new_result['feature_exp_min_mp'] = min(melting_points)
        new_result['feature_exp_max_mp'] = max(melting_points)
        new_result['feature_exp_mean_mp'] = np.mean(melting_points)
        new_result['feature_exp_div_mp'] = max(melting_points) - min(melting_points)
        
        
    target_names = [x["target"] for x in only_ss_rxns]
    precursors = [x["precursors"] for x in only_ss_rxns]
    to_predict = [np.mean(x[objective]) for x in only_ss_rxns]
    num_pts = [len(x[objective]) for x in only_ss_rxns]
    feature_exp_min_mp = [x['feature_exp_min_mp'] for x in only_ss_rxns]
    feature_exp_max_mp = [x['feature_exp_max_mp'] for x in only_ss_rxns]
    feature_exp_mean_mp = [x['feature_exp_mean_mp'] for x in only_ss_rxns]
    feature_exp_div_mp = [x['feature_exp_div_mp'] for x in only_ss_rxns]
    
    feature_calculators = MultipleFeaturizer([
        cf.ElementFraction()
    ])
    
    
    feature_labels = feature_calculators.feature_labels()
    
    data = pd.DataFrame()
    data["targets"] = target_names
    data["precursors"] = precursors
    data[objective] = to_predict
    df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(data, "targets", ignore_errors=True)
    df = feature_calculators.featurize_dataframe(df, col_id="composition_obj", ignore_errors = True)
    
    for key in ss_anion_dict:
        df[key + "_prec"] = ss_anion_dict[key]
        
    df['feature_exp_min_mp'] = feature_exp_min_mp
    df['feature_exp_max_mp'] = feature_exp_max_mp
    df['feature_exp_mean_mp'] = feature_exp_mean_mp
    df['feature_exp_div_mp'] = feature_exp_div_mp
    
    return df, num_pts

def add_pca_feats(only_ss_rxns, objective):
    target_names = [x["target"] for x in only_ss_rxns]
    precursors = [x["precursors"] for x in only_ss_rxns]
    to_predict = [np.mean(x[objective]) for x in only_ss_rxns]
    num_pts = [len(x[objective]) for x in only_ss_rxns]

    feature_calculators = MultipleFeaturizer([
        cf.ElementFraction()
    ])
    feature_labels = feature_calculators.feature_labels()

    data = pd.DataFrame()
    data["targets"] = target_names
    data["precursors"] = precursors
    data[objective] = to_predict
    df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(data, "targets", ignore_errors=True)
    df = feature_calculators.featurize_dataframe(df, col_id="composition_obj", ignore_errors = True)

    unique_precs = dict()
    for precs in df["precursors"].values:
        for prec in precs:
            if prec not in unique_precs:
                unique_precs[prec] = feature_calculators.featurize(Composition(prec))

    prec_magpie_feats = []
    train_precs = df["precursors"].values
    for precs in train_precs:
        feats = []
        for prec in precs:
            # check if nan
            if any(np.isnan(x) for x in unique_precs[prec]):
                feats.append(np.zeros(shape=(103)))
                continue
            feats.append(unique_precs[prec])
        for i in range(len(precs), 5):
            feats.append(np.zeros(shape=(103)))
        feats = np.array(feats)
        feats = feats.flatten()
        prec_magpie_feats.append(feats)
    prec_magpie_feats = np.array(prec_magpie_feats)

    return df, prec_magpie_feats, num_pts
    
def train_xgb(X, y, n_pts, featurization, prec_magpie_feats=None):
    # hyperparameter tuning
    # XGB hyperparameters

    n_estimators = [25, 50, 100, 200]
    max_depth=[3, 6, 9]
    learning_rate = [0.1, 0.2, 0.3]
    colsample_bytree = [0.6, 0.8, 1]
    colsample_bylevel = [0.6, 0.8, 1]
    min_child_weight = [1, 2, 3]
    subsample = [0.7, 0.9, 1]

    orig_parameters = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'colsample_bytree': colsample_bytree,
                'colsample_bylevel': colsample_bylevel,
                'min_child_weight': min_child_weight,
                'subsample': subsample
    }

    # Create the random grid
    parameters = {'n_estimators': hp.choice('n_estimators', n_estimators),
                'max_depth': hp.choice('max_depth', max_depth),
                'learning_rate': hp.choice('learning_rate', learning_rate),
                'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree),
                'colsample_bylevel': hp.choice('colsample_bylevel', colsample_bylevel),
                'min_child_weight': hp.choice('min_child_weight', min_child_weight),
                'subsample': hp.choice('subsample', subsample)
    }

    def objective(parameters):
        regr = XGBRegressor(
            n_estimators=parameters['n_estimators'],
            max_depth=parameters['max_depth'],
            learning_rate=parameters['learning_rate'],
            colsample_bytree=parameters['colsample_bytree'],
            colsample_bylevel=parameters['colsample_bylevel'],
            min_child_weight=parameters['min_child_weight'],
            subsample=parameters['subsample'],
            objective ="reg:squarederror"
        )

        regr.fit(X_train_curr, y_train_curr, eval_metric='rmse')

        y_pred_val = regr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        return {'loss': rmse, 'status': STATUS_OK}

    
    best_params = []
    best_estimators = []
    y_pred_train, y_pred_test = [], []
    X_train_k, X_test_k = [], []
    y_train_k, y_test_k = [], []
    n_pts_train_k, n_pts_test_k = [], []

    kf = KFold(n_splits=10, shuffle=False)
    start = time.time()

    for train_index, test_index in kf.split(X):
        # print((time.time() - start) / 60)
        # create train and test for this cv split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_pts_train, n_pts_test = n_pts[train_index], n_pts[test_index]
        if featurization == "pca":
            prec_magpie_feats_train, prec_magpie_feats_test = prec_magpie_feats[train_index], prec_magpie_feats[test_index]

        # create val split for this cv split from train
        # 72/18/10 split
        if featurization == "pca":
            X_train_curr, X_val, y_train_curr, y_val, prec_magpie_feats_train_curr, prec_magpie_feats_val = train_test_split(X_train, y_train, prec_magpie_feats_train, test_size=0.2, shuffle=False)
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = prec_scaler.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = prec_scaler.transform(prec_magpie_feats_val)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = pca.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = pca.transform(prec_magpie_feats_val)
            
            # concatenate with target features
            X_train_curr = np.concatenate((X_train_curr, prec_magpie_feats_train_curr), axis=1)
            X_val = np.concatenate((X_val, prec_magpie_feats_val), axis=1)

        else:
            X_train_curr, X_val, y_train_curr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        best_parameters = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=100)

        # convert index to value
        for key in best_parameters:
            best_parameters[key] = orig_parameters[key][best_parameters[key]]


        best_params.append(best_parameters)

        # get best estimator and get train and test predictions

        regr_best = XGBRegressor(
            n_estimators=best_parameters['n_estimators'],
            max_depth=best_parameters['max_depth'],
            learning_rate=best_parameters['learning_rate'],
            colsample_bytree=best_parameters['colsample_bytree'],
            colsample_bylevel=best_parameters['colsample_bylevel'],
            min_child_weight=best_parameters['min_child_weight'],
            subsample=best_parameters['subsample'],
            objective='reg:squarederror', 
        )

        if featurization == "pca":
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train = prec_scaler.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = prec_scaler.transform(prec_magpie_feats_test)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train= pca.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = pca.transform(prec_magpie_feats_test)
            
            # concatenate with target features
            X_train = np.concatenate((X_train, prec_magpie_feats_train), axis=1)
            X_test = np.concatenate((X_test, prec_magpie_feats_test), axis=1)

        regr_best.fit(X_train, y_train, eval_metric='rmse')

        best_estimators.append(regr_best)

        y_pred_train.append(regr_best.predict(X_train))
        y_pred_test.append(regr_best.predict(X_test))

        X_train_k.append(X_train)
        X_test_k.append(X_test)

        y_train_k.append(y_train)
        y_test_k.append(y_test)

        n_pts_train_k.append(n_pts_train)
        n_pts_test_k.append(n_pts_test)
        
    return best_params, best_estimators, y_pred_train, y_pred_test, X_train_k, X_test_k, y_train_k, y_test_k, n_pts_train_k, n_pts_test_k

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return torch.tensor(self.X_data[index], dtype=torch.float), torch.tensor(self.y_data[index], dtype=torch.float)
        
    def __len__ (self):
        return len(self.X_data)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, activation, layers_data, learning_rate=0.01):
        super().__init__()
        
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "tanh":
            activation = nn.Tanh()
        elif activation == "logistic":
            activation = nn.Sigmoid()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            self.layers.append(activation)
            
        # add final layer
        self.layers.append(nn.Linear(input_size, output_size))
        
        
        self.device = torch.device('cuda')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
    def fit(self, train_loader, val_loader=None, batch_size=256, num_epochs=500, validation=False, return_early_stopping=False):
        loss_stats = {
            'train': [],
            'val': []
        }
        

        
        early_stopping = EarlyStopping()
            
        for e in range(1, num_epochs+1):
        
            # TRAINING
            train_epoch_loss = 0

            self.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)

                self.optimizer.zero_grad()
                
                y_train_pred = self.forward(X_train_batch)
                

                train_loss = self.criterion(y_train_pred, y_train_batch)

                train_loss.backward()
                self.optimizer.step()

                train_epoch_loss += train_loss.item()
                
            loss_stats['train'].append(train_epoch_loss/len(train_loader))


            if validation:
                
                # VALIDATION    
                with torch.no_grad():

                    val_epoch_loss = 0

                    self.eval()
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                        y_val_pred = self.forward(X_val_batch)

                        val_loss = self.criterion(y_val_pred, y_val_batch)

                        val_epoch_loss += val_loss.item()

                
                loss_stats['val'].append(val_epoch_loss/len(val_loader))
                
#                 if (e-1) % 10 == 0:
#                     print(f'Epoch {e}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
                
                early_stopping(val_epoch_loss)
                if early_stopping.early_stop:
                    print(f'Epoch {e}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
                    if return_early_stopping:
                        return e
                    break
                    
            else:
#                 print(f'Epoch {e}: | Train Loss: {train_epoch_loss/len(train_loader):.5f}')
                  pass

        return e
        
    def predict(self, val_loader):


        # TEST
        with torch.no_grad():

            self.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                y_pred = self.forward(X_val_batch)

        return y_pred.cpu().numpy()
        # return np.array(y_val_pred).flatten()
                    


def train_nn(X, y, n_pts, featurization, prec_magpie_feats=None):
    # hyperparameter tuning
    # NN hyperparameters
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(y.shape[0], 1)

    hidden_layer_sizes = [
        [64, 128, 128, 64],
        [128, 256, 64],
        [64, 128, 32],
        [128, 64],
        [64, 32],
        [128],
        [64],
        [32]

    ]
    learning_rate = [
        0.0001,
        0.001,
        0.01
    ]
    activation = [
        'tanh',
        'relu',
        'logistic'
    ]

    orig_parameters = {'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'learning_rate': learning_rate,
    }

    # Create the random grid
    parameters = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', hidden_layer_sizes),
                'activation': hp.choice('activation', activation),
                'learning_rate': hp.choice('learning_rate', learning_rate),
    }

    def objective(parameters):
        print(parameters)
        regr = MLP(
            input_size=X_train_curr.shape[-1],
            output_size=1,
            activation=parameters['activation'],
            layers_data=parameters['hidden_layer_sizes'],
            learning_rate=parameters['learning_rate'],
    )
        regr.fit(train_loader, val_loader, validation=True)
        
        y_pred_val = regr.predict(val_loader)
        curr_rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        del regr
        
        return {'loss': curr_rmse, 'status': STATUS_OK}

    best_params = []
    best_estimators = []
    y_pred_train, y_pred_test = [], []
    X_train_k, X_test_k = [], []
    y_train_k, y_test_k = [], []
    n_pts_train_k, n_pts_test_k = [], []

    kf = KFold(n_splits=10, shuffle=False)
    start = time.time()

    for train_index, test_index in kf.split(X):
        print((time.time() - start) / 60)
        # create train and test for this cv split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_pts_train, n_pts_test = n_pts[train_index], n_pts[test_index]
        if featurization == "pca":
            prec_magpie_feats_train, prec_magpie_feats_test = prec_magpie_feats[train_index], prec_magpie_feats[test_index]
            X_train_curr, X_val, y_train_curr, y_val, prec_magpie_feats_train_curr, prec_magpie_feats_val = train_test_split(X_train, y_train, prec_magpie_feats_train, test_size=0.2, shuffle=False)

            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = prec_scaler.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = prec_scaler.transform(prec_magpie_feats_val)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = pca.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = pca.transform(prec_magpie_feats_val)
            
            # concatenate with target features
            X_train_curr = np.concatenate((X_train_curr, prec_magpie_feats_train_curr), axis=1)
            X_val = np.concatenate((X_val, prec_magpie_feats_val), axis=1)

        else:
            X_train_curr, X_val, y_train_curr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
        # create val split for this cv split from train
        # 72/18/10 split
        
        # scale features
        scaler = StandardScaler()
        scaler.fit(X_train_curr)
        X_train_curr = scaler.transform(X_train_curr)
        X_val = scaler.transform(X_val)   
            
        train_curr_data = ClassifierDataset(X_train_curr, y_train_curr)
        val_data = ClassifierDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
        dataset=train_curr_data,
        batch_size=256,
        shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=X_val.shape[0],
        shuffle=False)
        
        best_parameters = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=10)
        
        # convert index to value
        for key in best_parameters:
            best_parameters[key] = orig_parameters[key][best_parameters[key]]
        
        best_params.append(best_parameters)
        
        # get best number of epochs to train
        regr_best = MLP(
            input_size=X_train_curr.shape[-1],
            output_size=1,
            activation=best_parameters['activation'],
            layers_data=best_parameters['hidden_layer_sizes'],
            learning_rate=best_parameters['learning_rate'],
    )
        
        best_n_epochs = regr_best.fit(train_loader, val_loader, validation=True, return_early_stopping=True)

        del regr_best
        
        # get best estimator and get train and test predictions
        
        regr_best = MLP(
            input_size=X_train_curr.shape[-1],
            output_size=1,
            activation=best_parameters['activation'],
            layers_data=best_parameters['hidden_layer_sizes'],
            learning_rate=best_parameters['learning_rate'],
    )
                
        if featurization == "pca":
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train = prec_scaler.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = prec_scaler.transform(prec_magpie_feats_test)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train= pca.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = pca.transform(prec_magpie_feats_test)
            
            # concatenate with target features
            X_train = np.concatenate((X_train, prec_magpie_feats_train), axis=1)
            X_test = np.concatenate((X_test, prec_magpie_feats_test), axis=1)
        
        # scale features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_data = ClassifierDataset(X_train, y_train)
        test_data = ClassifierDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=256,
        shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=X_test.shape[0],
        shuffle=False)
        
        regr_best.fit(train_loader, val_loader, num_epochs=best_n_epochs, validation=False)
        
        best_estimators.append(regr_best)
        
        y_pred_train.append(regr_best.predict(train_loader))
        y_pred_test.append(regr_best.predict(val_loader))
        
        X_train_k.append(scaler.inverse_transform(X_train))
        X_test_k.append(scaler.inverse_transform(X_test))
        
        y_train_k.append(y_train)
        y_test_k.append(y_test)

        n_pts_train_k.append(n_pts_train)
        n_pts_test_k.append(n_pts_test)

        del regr_best
        
    return best_params, best_estimators, y_pred_train, y_pred_test, X_train_k, X_test_k, y_train_k, y_test_k, n_pts_train_k, n_pts_test_k

def train_rf(X, y, n_pts, featurization, prec_magpie_feats=None):
    n_estimators = [25, 50, 100, 200]
    max_features = ['auto', 'log2']
    max_depth=[10, 20, 30, 50, 100, None]
    min_samples_split = [2, 3, 5, 10]
    min_samples_leaf = [1, 2, 3, 5]
    bootstrap = [True, False]

    orig_parameters = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    parameters = {'n_estimators': hp.choice('n_estimators', n_estimators),
                'max_features': hp.choice('max_features', max_features),
                'max_depth': hp.choice('max_depth', max_depth),
                'min_samples_split': hp.choice('min_samples_split', min_samples_split),
                'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf),
                'bootstrap': hp.choice('bootstrap', bootstrap)}

    def objective(parameters):
        regr = RandomForestRegressor(
            n_estimators=parameters['n_estimators'],
            max_features=parameters['max_features'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
            min_samples_leaf=parameters['min_samples_leaf'],
            bootstrap=parameters['bootstrap']
        )

        regr.fit(X_train_curr, y_train_curr)

        y_pred_val = regr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        return {'loss': rmse, 'status': STATUS_OK}

    
    best_params = []
    best_estimators = []
    y_pred_train, y_pred_test = [], []
    X_train_k, X_test_k = [], []
    y_train_k, y_test_k = [], []
    n_pts_train_k, n_pts_test_k = [], []

    kf = KFold(n_splits=10, shuffle=False)
    start = time.time()

    for train_index, test_index in kf.split(X):
        # print((time.time() - start) / 60)
        # create train and test for this cv split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_pts_train, n_pts_test = n_pts[train_index], n_pts[test_index]
        if featurization == "pca":
            prec_magpie_feats_train, prec_magpie_feats_test = prec_magpie_feats[train_index], prec_magpie_feats[test_index]

        # create val split for this cv split from train
        # 72/18/10 split
        if featurization == "pca":
            X_train_curr, X_val, y_train_curr, y_val, prec_magpie_feats_train_curr, prec_magpie_feats_val = train_test_split(X_train, y_train, prec_magpie_feats_train, test_size=0.2, shuffle=False)
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = prec_scaler.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = prec_scaler.transform(prec_magpie_feats_val)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = pca.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = pca.transform(prec_magpie_feats_val)
            
            # concatenate with target features
            X_train_curr = np.concatenate((X_train_curr, prec_magpie_feats_train_curr), axis=1)
            X_val = np.concatenate((X_val, prec_magpie_feats_val), axis=1)

        else:
            X_train_curr, X_val, y_train_curr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        best_parameters = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=50)

        # convert index to value
        for key in best_parameters:
            best_parameters[key] = orig_parameters[key][best_parameters[key]]


        best_params.append(best_parameters)

        # get best estimator and get train and test predictions

        regr_best = RandomForestRegressor(
            n_estimators=best_parameters['n_estimators'],
            max_features=best_parameters['max_features'],
            max_depth=best_parameters['max_depth'],
            min_samples_split=best_parameters['min_samples_split'],
            min_samples_leaf=best_parameters['min_samples_leaf'],
            bootstrap=best_parameters['bootstrap'],
        )

        if featurization == "pca":
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train = prec_scaler.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = prec_scaler.transform(prec_magpie_feats_test)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train= pca.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = pca.transform(prec_magpie_feats_test)
            
            # concatenate with target features
            X_train = np.concatenate((X_train, prec_magpie_feats_train), axis=1)
            X_test = np.concatenate((X_test, prec_magpie_feats_test), axis=1)

        regr_best.fit(X_train, y_train)

        best_estimators.append(regr_best)

        y_pred_train.append(regr_best.predict(X_train))
        y_pred_test.append(regr_best.predict(X_test))

        X_train_k.append(X_train)
        X_test_k.append(X_test)

        y_train_k.append(y_train)
        y_test_k.append(y_test)

        n_pts_train_k.append(n_pts_train)
        n_pts_test_k.append(n_pts_test)
        
    return best_params, best_estimators, y_pred_train, y_pred_test, X_train_k, X_test_k, y_train_k, y_test_k, n_pts_train_k, n_pts_test_k

def train_lr(X, y, n_pts, featurization, prec_magpie_feats=None):
    alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Create the random grid

    orig_parameters = {'alpha': alpha,
                'l1_ratio': l1_ratio
    }

    parameters = {'alpha': hp.choice('alpha', alpha),
                'l1_ratio': hp.choice('l1_ratio', l1_ratio)
    }

    def objective(parameters):
        regr = ElasticNet(
            alpha=parameters['alpha'],
            l1_ratio=parameters['l1_ratio'],
            max_iter=1000
        )

        regr.fit(X_train_curr, y_train_curr)

        y_pred_val = regr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)

        return {'loss': rmse, 'status': STATUS_OK}

    best_params = []
    best_estimators = []
    y_pred_train, y_pred_test = [], []
    X_train_k, X_test_k = [], []
    y_train_k, y_test_k = [], []
    n_pts_train_k, n_pts_test_k = [], []

    kf = KFold(n_splits=10, shuffle=False)
    start = time.time()

    for train_index, test_index in kf.split(X):
        # print((time.time() - start) / 60)
        # create train and test for this cv split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_pts_train, n_pts_test = n_pts[train_index], n_pts[test_index]
        if featurization == "pca":
            prec_magpie_feats_train, prec_magpie_feats_test = prec_magpie_feats[train_index], prec_magpie_feats[test_index]

        # create val split for this cv split from train
        # 72/18/10 split
        if featurization == "pca":
            X_train_curr, X_val, y_train_curr, y_val, prec_magpie_feats_train_curr, prec_magpie_feats_val = train_test_split(X_train, y_train, prec_magpie_feats_train, test_size=0.2, shuffle=False)
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = prec_scaler.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = prec_scaler.transform(prec_magpie_feats_val)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train_curr)
            # prec_magpie_feats_train_curr = pca.transform(prec_magpie_feats_train_curr)
            # prec_magpie_feats_val = pca.transform(prec_magpie_feats_val)
            
            # concatenate with target features
            X_train_curr = np.concatenate((X_train_curr, prec_magpie_feats_train_curr), axis=1)
            X_val = np.concatenate((X_val, prec_magpie_feats_val), axis=1)

        else:
            X_train_curr, X_val, y_train_curr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        # scale features
        scaler = StandardScaler()
        scaler.fit(X_train_curr)
        X_train_curr = scaler.transform(X_train_curr)
        X_val = scaler.transform(X_val) 

        best_parameters = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=50)

        # convert index to value
        for key in best_parameters:
            best_parameters[key] = orig_parameters[key][best_parameters[key]]


        best_params.append(best_parameters)

        # get best estimator and get train and test predictions

        regr_best = ElasticNet(
            alpha=best_parameters['alpha'],
            l1_ratio=best_parameters['l1_ratio'],
            max_iter=1000
        )

        if featurization == "pca":
            # scale prec feats and PCA compress to 10 dimensions
            # prec_scaler = StandardScaler()
            # prec_scaler.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train = prec_scaler.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = prec_scaler.transform(prec_magpie_feats_test)
            
            # pca = PCA(n_components=10)
            # pca = pca.fit(prec_magpie_feats_train)
            # prec_magpie_feats_train= pca.transform(prec_magpie_feats_train)
            # prec_magpie_feats_test = pca.transform(prec_magpie_feats_test)
            
            # concatenate with target features
            X_train = np.concatenate((X_train, prec_magpie_feats_train), axis=1)
            X_test = np.concatenate((X_test, prec_magpie_feats_test), axis=1)

        # scale features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        regr_best.fit(X_train, y_train)

        best_estimators.append(regr_best)

        y_pred_train.append(regr_best.predict(X_train))
        y_pred_test.append(regr_best.predict(X_test))

        X_train_k.append(X_train)
        X_test_k.append(X_test)

        y_train_k.append(y_train)
        y_test_k.append(y_test)

        n_pts_train_k.append(n_pts_train)
        n_pts_test_k.append(n_pts_test)
        
    return best_params, best_estimators, y_pred_train, y_pred_test, X_train_k, X_test_k, y_train_k, y_test_k, n_pts_train_k, n_pts_test_k


