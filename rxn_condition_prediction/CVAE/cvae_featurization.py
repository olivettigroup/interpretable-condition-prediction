import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pymatgen import Composition, Element
import pickle
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, precision_score, r2_score, mean_absolute_error
from pymatgen import Composition, Element
from sklearn.utils import shuffle
from cvae import TempTimeGenerator
from tqdm import tqdm
import json
import sys, os
from hyperopt import fmin, tpe, hp, STATUS_OK
sys.path.append('/home/jupyter/CJK/TempTime')
from data_extraction_rxn_classification_all_targets import *

def impute_data(imputation_strategy):
    assert imputation_strategy in ["one_hot", "pca", "mp_fraction"]

    with open('/home/jupyter/CJK/TempTime/data/solid-state_dataset_20200713.json') as f:
        ss_data = json.load(f)
    ss_reactions = ss_data["reactions"]
    ss_extracted, ss_precursor_nomenclature = extract_solidstate(ss_reactions, max_pre=5)

    corpus = ss_extracted
    corpus_to_extract = 1

    # feature_calculators = MultipleFeaturizer([
    #     cf.ElementFraction()
    # ])

    unique_precs_dict = dict()
    for rxn in corpus:
        for prec in rxn['precursors']:
            comp = Composition(prec)
            prec_formula = comp.reduced_formula
            if prec_formula not in unique_precs_dict:
                # unique_precs_dict[prec_formula] = feature_calculators.featurize(comp)
                unique_precs_dict[prec_formula] = prec_formula

    unique_precs = list(unique_precs_dict.keys())

    with open('/home/jupyter/CJK/TempTime/rxn_classification_fraction/data/full_mp.json', 'r') as f:
        full_mp = json.load(f)
        
    full_mp = {Composition(x): full_mp[x] for x in full_mp}
    full_mp_median = np.median(list(full_mp.values()))

    all_papers = []
    full_papers = []
    # operation tokens are in order of precedence found
    # temp_time_ops = [["calcin", "fire", "heat"], ["sinter"], ["anneal"], ["dry", "dried"]]
    temp_time_ops = [["calcin"], ["sinter"], ["anneal"], ["dry", "dried"]]
    bad_re = ['refired', 're-fired', 're-sintered', 'resintered' 'recalcined', 're-calcined', 're-annealed', 'reannealed',
            'reannealing', 're-heated', 'reheated', 'reheating']
    ss_data, sg_data = [], []


    if corpus_to_extract == 0:
        # remove heat since it's ambiguous in sol-gel
        #pass
        temp_time_ops[0] = temp_time_ops[0][:2]


    for num, processed_paper in enumerate(tqdm(corpus)):
        # check if has at least one temp/time in reverse
        temp_conditions = np.empty(len(temp_time_ops))
        time_conditions = np.empty(len(temp_time_ops))
        temp_conditions[:] = np.nan
        time_conditions[:] = np.nan
        for i, phrases in enumerate(temp_time_ops):
            found = False
            for phrase in phrases:
                if not found:
                    for j, op in list(enumerate(processed_paper['operation_tokens']))[::-1]:
                        if phrase in op and 'pre' not in op and 'post' not in op and phrase not in bad_re and processed_paper['operation_temps'][j] and ((i <= 1 and processed_paper['operation_temps'][j] >= 100) or (i > 1 and processed_paper['operation_temps'][j] > 0)):
                            temp_conditions[i] = processed_paper['operation_temps'][j]
                            if processed_paper['operation_times'][j] and processed_paper['operation_times'][j] > 0:
                                time_conditions[i] = processed_paper['operation_times'][j]
                            found = True
                            break

        # check if we have at least one temperature to report, excluding the maximum temperature
        # if np.isnan(temp_conditions).all():
        #     continue
        # check if we have at least sintering OR calcination temperature to report
        if np.isnan(temp_conditions[0]) and np.isnan(temp_conditions[1]):
            continue
        # featurize precs
        if imputation_strategy == "one_hot":
            prec_vector = np.zeros(len(unique_precs))
            for prec in processed_paper['precursors']:
                prec_formula = Composition(prec).reduced_formula
                prec_vector[unique_precs.index(prec_formula)] = 1.0
        elif imputation_strategy == "pca" or imputation_strategy == "mp_fraction":
            prec_vector = featurize_precursors(processed_paper['precursors'], imputation_strategy, ss_precursor_nomenclature, full_mp, full_mp_median)
            prec_vector = prec_vector.astype(float)

        temp_time_rxn = np.concatenate((temp_conditions, time_conditions, prec_vector))
        # all papers is before imputation
        all_papers.append(temp_time_rxn)
        # full papers is the original paper objects
        full_papers.append(processed_paper)

    all_papers = np.array(all_papers)

    mins = [-np.inf for x in range(all_papers[0, 8:].shape[-1])]
    maxes = [np.inf for x in range(all_papers[0, 8:].shape[-1])]

    min_value = np.zeros(8)
    max_value = np.zeros(8)

    for i in range(8):
        min_value[i] = min([x[i] for x in all_papers if not np.isnan(x[i])])
        max_value[i] = max([x[i] for x in all_papers if not np.isnan(x[i])])
    
    min_value = np.concatenate([min_value, mins], axis=-1)
    max_value = np.concatenate([max_value, maxes], axis=-1)

    imp_mean = IterativeImputer(max_iter=10, sample_posterior=True, min_value=min_value, max_value=max_value, skip_complete=True, random_state=42)
    # imputed_papers = imp_mean.fit_transform(all_papers)
    imp_mean.fit(all_papers)
    imputed_papers = imp_mean.transform(all_papers)

    for i in range(len(full_papers)):
        full_papers[i]["temp_time_vector"] = imputed_papers[i][:8]
    
    return full_papers, ss_precursor_nomenclature



def featurize(featurization, only_ss_rxns, ss_precursor_nomenclature):

    assert featurization in ["pca", "mp_fraction"]

    objective = 'temp_time_vector'

    targets = [x['target'] for x in only_ss_rxns]
    precursors = [x['precursors'] for x in only_ss_rxns]
    to_predict = [x[objective] for x in only_ss_rxns]
    dois = [x['DOI'] for x in only_ss_rxns]

    # change precursor functional groups back
    for rxn in only_ss_rxns:
        for i in range(len(rxn['precursors'])):
            rxn['precursors'][i] = ss_precursor_nomenclature[Composition(rxn['precursors'][i]).reduced_composition]
            
    if featurization=="mp_fraction":
        df, prec_magpie_feats = add_mp_fraction_feats(only_ss_rxns, objective)
    elif featurization=="pca":
       df, prec_magpie_feats = add_pca_feats(only_ss_rxns, objective)

    return df, prec_magpie_feats

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
    to_predict = [x[objective] for x in only_ss_rxns]

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

    prec_df = pd.DataFrame()
    
    for key in ss_anion_dict:
        prec_df[key + "_prec"] = ss_anion_dict[key]
        
    prec_df['feature_exp_min_mp'] = feature_exp_min_mp
    prec_df['feature_exp_max_mp'] = feature_exp_max_mp
    prec_df['feature_exp_mean_mp'] = feature_exp_mean_mp
    prec_df['feature_exp_div_mp'] = feature_exp_div_mp
    
    return df, prec_df

def add_pca_feats(only_ss_rxns, objective):
    target_names = [x["target"] for x in only_ss_rxns]
    precursors = [x["precursors"] for x in only_ss_rxns]
    to_predict = [x[objective] for x in only_ss_rxns]

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

    return df, prec_magpie_feats

def featurize_precursors(precursors, featurization_strategy, ss_precursor_nomenclature, full_mp, full_mp_median):
    if featurization_strategy == "pca":
        feature_calculators = MultipleFeaturizer([
            cf.ElementFraction()
        ])

        feats = []
        for prec in precursors:
            feat = feature_calculators.featurize(Composition(prec))
            # check if nan
            if any(np.isnan(x) for x in feat):
                feats.append(np.zeros(shape=(103)))
                continue
            feats.append(feat)
        for i in range(len(precursors), 5):
            feats.append(np.zeros(shape=(103)))
        feats = np.array(feats)
        feats = feats.flatten()
        return feats
    elif featurization_strategy == "mp_fraction":

        # change precursor functional groups back
        for i in range(len(precursors)):
            precursors[i] = ss_precursor_nomenclature[Composition(precursors[i]).reduced_composition]

        feats = []

        all_elements = [Element.from_Z(i).symbol for i in range(1, 119)]
                
        # Onehot precursor functional groups
        m_anions = ['H2PO4', 'HPO4', 'HCO3', 'HSO4', 'HSO3', 'C2O4']
        d_anions = ['CO3', 'PO4', 'PO3', 'OH', 'NH4', 'NO3', 'NO2', 'SO4', 'SO3', 'CN'] #BO3, VO4, NH2
        s_anions = ['F', 'B', 'P', 'Cl', 'F', 'Br', 'S', 'N', 'O', 'C']
        ions = m_anions + d_anions + s_anions + ['Org'] + ['Ac'] + ['Elem'] + ['Other']
        
        anion_dict = {x: [] for x in ions}
        for prec in precursors:
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
        fractional_feats = []

        for key in anion_dict:
            fractional_feats.append(len(anion_dict[key]) / len(precursors))

        feats.append(fractional_feats)

        melting_points = [full_mp.get(Composition(x), full_mp_median) for x in precursors]

        feats.append(min(melting_points))
        feats.append(max(melting_points))
        feats.append(np.mean(melting_points))
        feats.append(max(melting_points) - min(melting_points))

        feats = np.hstack(feats)
        return feats

def get_unique_test_set(df):
    test_results = []
    unique_inds = []
    for i, (target, precs, temps) in enumerate(zip(df['targets'], df['precursors'], df['temp_time_vector'])):
        found = False
        for result in test_results:
            if result["Target"] == target and set(result["Precursors"]) == set(precs):
                result["temp_time_vector"].append(temps)
                found = True
        if not found:
            new_result = {}
            new_result["Target"] = target
            new_result["Precursors"] = precs
            new_result["temp_time_vector"] = [temps]
            test_results.append(new_result)
            unique_inds.append(i)
    return test_results, np.array(unique_inds)

def train(X, y, prec_magpie_feats, full_df):
    # hyperparameter tuning
    rnn_dim = [16, 32, 64]
    conv_filters = [8, 16, 32]
    intermediate_dim = [64, 128, 256]
    latent_dim = [3, 4, 5, 6]

    orig_parameters = {
        'rnn_dim': rnn_dim,
        'conv_filters': conv_filters,
        'intermediate_dim': intermediate_dim,
        'latent_dim': latent_dim
    }

    # create the random grid
    parameters = {
        'rnn_dim': hp.choice('rnn_dim', rnn_dim),
        'conv_filters': hp.choice('conv_filters', conv_filters),
        'intermediate_dim': hp.choice('intermediate_dim', intermediate_dim),
        'latent_dim': hp.choice('latent_dim', latent_dim)
    }

    def objective(parameters):
        temp_gen = TempTimeGenerator()
        temp_gen.build_nn_model(rnn_dim=parameters['rnn_dim'], 
        conv_filters=parameters['conv_filters'],
        intermediate_dim=parameters['intermediate_dim'],
        latent_dim=parameters['latent_dim'],
        precursor_len=prec_magpie_feats.shape[-1])

        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')]
        history = temp_gen.train(
            inputs=train_inputs, 
            outputs=train_outputs,
            epochs=500,
            val_split=0,
            val_data=val_data,
            batch_size=128,
            callbacks=callbacks,
            verbose=2
        )

        # return {'loss': history['val_loss'][-1], 'status': STATUS_OK}
        results = []
        for i in unique_inds_val:
            conds = temp_gen.generate_samples(target_material=X_val[i:i+1], precursors=prec_magpie_feats_val[i:i+1], n_samples=100)
            curr_results = []
            for conditions in conds:
                temp_time = scaler.inverse_transform(conditions.reshape(1, -1)).flatten()
                curr_results.append(temp_time)
            curr_results = np.vstack(curr_results)
            results.append(curr_results)
        results = np.array(results)

        y_pred = np.mean(results, axis=1)
        y_true = np.vstack([np.mean(x['temp_time_vector'], axis=0) for x in val_results])
        rmse_to_minimize = []
        for i in [0, 1, 3, 4]:
            rmse_to_minimize.append(mean_squared_error(y_true[:, i], y_pred[:, i], squared=False))

        del temp_gen
 

        return {'loss': np.mean(rmse_to_minimize), 'status': STATUS_OK}

    best_params = []
    y_pred_train, y_pred_test = [], []
    X_train_k, X_test_k = [], []
    y_train_k, y_test_k = [], []
    X_train_df, X_test_df = [], []
    n_pts_train_k, n_pts_test_k = [], []
    histories = []

    kf = KFold(n_splits=10, shuffle=False)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        df_train, df_test = full_df.iloc[train_index], full_df.iloc[test_index]
        
        prec_magpie_feats_train, prec_magpie_feats_test = prec_magpie_feats[train_index], prec_magpie_feats[test_index]
        X_train_curr, X_val, y_train_curr, y_val, prec_magpie_feats_train_curr, prec_magpie_feats_val, _, df_val = train_test_split(X_train, y_train, prec_magpie_feats_train, df_train, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        scaler.fit(X_train_curr)
        X_train_curr = scaler.transform(X_train_curr)
        X_val = scaler.transform(X_val)
        
        scaler = StandardScaler()
        scaler.fit(prec_magpie_feats_train_curr)
        prec_magpie_feats_train_curr = scaler.transform(prec_magpie_feats_train_curr)
        prec_magpie_feats_val = scaler.transform(prec_magpie_feats_val)
        
        scaler = StandardScaler()
        scaler.fit(y_train_curr)
        y_train_curr = scaler.transform(y_train_curr)
        y_val = scaler.transform(y_val)
        
        y_train_curr = np.reshape(y_train_curr, (-1, 8, 1))
        y_val = np.reshape(y_val, (-1, 8, 1))
        
        train_inputs = [y_train_curr, X_train_curr, prec_magpie_feats_train_curr]
        train_outputs = [y_train_curr]
        
        val_data = [[y_val, X_val, prec_magpie_feats_val], y_val]

        # hyperparameter search
        val_results, unique_inds_val = get_unique_test_set(df_val)

        best_parameters = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=20)

        # convert index to value
        for key in best_parameters:
            best_parameters[key] = orig_parameters[key][best_parameters[key]]
        
        temp_gen = TempTimeGenerator()
        temp_gen.build_nn_model(
            rnn_dim=best_parameters['rnn_dim'],
            conv_filters=best_parameters['conv_filters'],
            intermediate_dim=best_parameters['intermediate_dim'],
            latent_dim=best_parameters['latent_dim'],
            precursor_len=prec_magpie_feats.shape[-1])
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')]
        history = temp_gen.train(
            inputs=train_inputs, 
            outputs=train_outputs,
            epochs=500,
            val_split=0,
            val_data=val_data,
            batch_size=128,
            callbacks=callbacks,
            verbose=2
        )
        
        histories.append(history)
        
        epochs = len(history['val_loss'])
        
        best_params.append([best_parameters, epochs])
        
        del temp_gen
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        scaler = StandardScaler()
        scaler.fit(prec_magpie_feats_train)
        prec_magpie_feats_train = scaler.transform(prec_magpie_feats_train)
        prec_magpie_feats_test = scaler.transform(prec_magpie_feats_test)
        
        scaler = StandardScaler()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_train = np.reshape(y_train, (-1, 8, 1))
        
        train_inputs = [y_train, X_train, prec_magpie_feats_train]
        train_outputs = [y_train]
        
        temp_gen = TempTimeGenerator()
        temp_gen.build_nn_model(
            rnn_dim=best_parameters['rnn_dim'],
            conv_filters=best_parameters['conv_filters'],
            intermediate_dim=best_parameters['intermediate_dim'],
            latent_dim=best_parameters['latent_dim'],
            precursor_len=prec_magpie_feats.shape[-1])

        history = temp_gen.train(
            inputs=train_inputs, 
            outputs=train_outputs,
            epochs=epochs,
            val_split=0,
            val_data=None,
            batch_size=128,
            callbacks=None,
            verbose=2
        )
        
        test_results, unique_inds = get_unique_test_set(df_test)
        
        results = []
        for i in unique_inds:
            conds = temp_gen.generate_samples(target_material=X_test[i:i+1], precursors=prec_magpie_feats_test[i:i+1], n_samples=1000)
            curr_results = []
            for conditions in conds:
                temp_time = scaler.inverse_transform(conditions.reshape(1, -1)).flatten()
                curr_results.append(temp_time)
            curr_results = np.vstack(curr_results)
            results.append(curr_results)
        results = np.array(results)
        
        y_pred_test.append(results)
        y_test_k.append(test_results)

        del temp_gen

        X_train_k.append(X_train)
        X_test_k.append(X_test)

        y_train_k.append(y_train)
        # y_test_k.append(y_test)

        X_train_df.append(df_train)
        X_test_df.append(df_test)

    return best_params, y_pred_train, y_pred_test, X_train_k, X_test_k, y_train_k, y_test_k, n_pts_train_k, n_pts_test_k, histories, X_train_df, X_test_df

def evaluate(imputation_strategy, featurization, y_pred_test, y_test_k):
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
    results = []
    for key in mapping:
        i = mapping[key]
        maes, rmses, mres, r2s = [], [], [], []
        contained, pred_contained = [], []
        for fold in range(10):
            y_pred = np.mean(y_pred_test[fold], axis=1)
            y_true = np.vstack([np.mean(x['temp_time_vector'], axis=0) for x in y_test_k[fold]])
            rmses.append(mean_squared_error(y_true[:, i], y_pred[:, i], squared=False))
            maes.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
            mres.append(np.mean((np.abs(y_true[:, i]-y_pred[:, i])/y_true[:, i])*100))
            r2s.append(r2_score(y_true[:, i], y_pred[:, i]))
            
            num_contained = 0
            num_pred_contained = 0
            num_pred = 0
            true_vals = [x['temp_time_vector'] for x in y_test_k[fold]]
            for ind in range(y_true.shape[0]):
                if min(y_pred_test[fold][ind, :, i]) <= y_true[ind][i] <= max(y_pred_test[fold][ind, :, i]):
                    num_contained += 1
                if len(true_vals[ind]) > 5:
                    exp_min = min(np.vstack(true_vals[ind])[:, i])
                    exp_max = max(np.vstack(true_vals[ind])[:, i])
                    num_pred += 1
                    if exp_min <= y_pred[ind, i] <= exp_max:
                        num_pred_contained += 1
                        
            pred_contained.append(num_pred_contained / num_pred)
            contained.append(num_contained / y_true.shape[0])
                    
        result = {
            "objective": key,
            "model": "CVAE",
            "imputation_strategy": imputation_strategy,
            "featurization": featurization,
            "MAE": str(np.mean(maes)),
            "MAE_std": str(np.std(maes)),
            "RMSE": str(np.mean(rmses)),
            "RMSE_std": str(np.std(rmses)),
            "MRE": str(np.mean(mres)),
            "MRE_std": str(np.std(mres)),
            "R2": str(np.mean(r2s)),
            "R2_std": str(np.std(r2s)),
            "Percent_exp_means_contained": str(np.mean(contained) * 100),
            "Percent_exp_means_contained_std": str(np.std(contained) * 100),
            "Percent_pred_means_contained": str(np.mean(pred_contained) * 100),
            "Percent_pred_means_contained_std": str(np.std(pred_contained) * 100)
        }
        results.append(result)

    with open('data/' + imputation_strategy + '_' + featurization + "_3_layers_sinter_calcine.json", "w") as f:
        json.dump(results, f, indent=4)
    return results


    