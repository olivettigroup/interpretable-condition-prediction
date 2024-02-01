import numpy as np
import pandas as pd
from pymatgen import Composition, Element
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

from pymatgen import Structure
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
mp = MPDataRetrieval(api_key='ADLdej8mhb8Pb92kpeVU')

# Featurize the compounds with compositional features
def featurize_compositional(basis: str, compound_list: list, label_prefix: str):
    if basis == 'magpie145':
        feature_calculators = MultipleFeaturizer([
        cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=["avg"]),
        cf.IonProperty(fast=True)
        ])

    if basis == 'deml':
        feature_calculators = MultipleFeaturizer([
        cf.ElementProperty.from_preset("deml"),
        cf.IonProperty(fast=True)
        ])

    if basis == 'subset1':
        data_source = 'magpie'
        features = ["Number", "MendeleevNumber", "AtomicWeight","MeltingT","Column", "Row", 
                    "CovalentRadius", "Electronegativity"]
        stats = ["minimum", "maximum"]
        feature_calculators = MultipleFeaturizer([
        cf.ElementProperty(data_source, features, stats),
        cf.IonProperty(fast=True)
        ])
        
    feature_labels = feature_calculators.feature_labels()

    data = pd.DataFrame()
    data[label_prefix] = compound_list
    df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(data, label_prefix, ignore_errors=True)
    df = feature_calculators.featurize_dataframe(df, col_id="composition_obj", ignore_errors = True)

    return df, feature_labels

# Featurize compounds as reactions
def featurize_compounds(reactions_df, features_df, column_mapping, suffixes=('','')):
    features_df = features_df.rename(columns={column_mapping[0]: column_mapping[1]})
    reactions_with_features = reactions_df.merge(features_df, how='inner', on=column_mapping[1], suffixes=suffixes)
    return reactions_with_features

# Featurize reactions 
def featurize_reactions(reactions_df, precursor_features, target_features):
    #print(len(reactions_df))
    reactions_with_features1 = featurize_compounds(reactions_df, precursor_features, ('Precursor','Prec1'))
    #print(len(reactions_with_features1))
    reactions_with_features2 = featurize_compounds(reactions_with_features1, precursor_features, ('Precursor','Prec2'), ('_Prec1','_Prec2'))
    #print(len(reactions_with_features2))
    reactions_with_features3 = featurize_compounds(reactions_with_features2, target_features, ('Target','target'), ('_Prec2','_Target'))
    #print(len(reactions_with_features3))
    #Remove columns that are all 0
    df = reactions_with_features3
    return df
# Only featurize targets
def featurize_only_targets(reactions_df, target_features):
    reactions_with_target_features = featurize_compounds(reactions_df, target_features, ('Target','target'), ('_Prec2','_Target'))
    return reactions_with_target_features

# Get X and y arrays for regression task
def get_Xy(features_df):
    omit_columns = ['DOI', 'target', 'precursors', 'temp', 'time', 'token', 'heat_steps_no', 'Prec1',
                    'Prec2', 'composition_obj_Prec1', 'composition_obj_Prec2','Structure_Prec1', 
                    'composition_obj_Target','Structure', 'Structure_Prec2', 'Structure_Target',
                    'composition_obj', 'PrecTargetPair', 'Rank']
    X_columns = list(set(list(features_df.columns)) - set(omit_columns))
    y_column = 'temp'
    features_df = features_df.dropna(axis=1, how='all')
    features_df = features_df.dropna(axis=0, how='any')
    X_df = features_df[X_columns]
    X_columns = list(X_df.columns)
    X = X_df.values
    y = features_df[y_column].values
    print("Shape of X: {}".format(X.shape))
    print("Shape of y: {}".format(y.shape))
    return X, y, X_columns, features_df

# Onehot the anions in the precursor materials
def featurize_anion_onehot(database, all_precs=True, agglomerate=True, return_df=True):
    first_pass_counterions = ['NO3', 'HCO3', 'PO3', 'H2PO4', 'HPO4' 'OH', 'C2O4', 'CH3COO', 'SO4', 'NH4']
    second_pass_counterions = ['PO4', 'CO3']
    third_pass_counterions = ['O', 'B', 'C', 'N', 'S', 'F', 'Cl', 'I', 'Br', 'P', 'Se', 'As', 'Te', 'H', 'Fe', 'Sb', 'Pb', 'Ni', 'Ga', 'Cu', 'Mg', 'Si', 'pure']
    counterions = first_pass_counterions + second_pass_counterions + third_pass_counterions
    counterion_arr = [first_pass_counterions, second_pass_counterions, third_pass_counterions]
    others = ['Fe', 'Sb', 'Pb', 'Ni', 'Ga', 'Cu', 'Mg', 'Si']
    outliers = ['Am0.07', 'Sm0.005']
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    
    if agglomerate:
        all_precs_onehot = []
        for item in database:
            onehot_prec = np.zeros(shape=len(counterions))
            for prec in item['precursors']:
                found = False
                # check if pure element
                if prec in all_elements:
                    onehot_prec[counterions.index('pure')] = 1.0
                    found = True
                # check in decreasing order of complexity
                for j, arr in enumerate(counterion_arr):
                    # for complex, just search regularly
                    if j < 2: 
                        for ion in arr:
                            if not found and ion in prec:
                                onehot_prec[counterions.index(ion)] = 1.0
                                found = True
                    # for simple, split into elements
                    else:
                        try:
                            prec_elements = [e.symbol for e in Composition(prec).elements]
                        except:
                            print(prec)
                        for ion in arr:
                            if not found and ion in prec_elements:
                                onehot_prec[counterions.index(ion)] = 1.0
                                found = True
            all_precs_onehot.append(onehot_prec)
        if return_df:
            return pd.DataFrame(all_precs_onehot, columns=counterions)
        else:
            return all_precs_onehot
    else:
        full_array = []
        all_precs_onehot = []
        for item in database:
            prec_array = []
            for prec in item['precursors']:
                found = False
                onehot_prec = np.zeros(shape=len(counterions))
                # check if pure element
                if prec in all_elements:
                    onehot_prec[counterions.index('pure')] = 1.0
                    found = True
                # check in decreasing order of complexity
                for j, arr in enumerate(counterion_arr):
                    # for complex, just search regularly
                    if j < 2: 
                        for ion in arr:
                            if not found and ion in prec:
                                onehot_prec[counterions.index(ion)] = 1.0
                                found = True
                    # for simple, split into elements
                    else:
                        prec_elements = [e.symbol for e in Composition(prec).elements]
                        for ion in arr:
                            if not found and ion in prec_elements:
                                onehot_prec[counterions.index(ion)] = 1.0
                                found = True
                prec_array.append(onehot_prec)
                all_precs_onehot.append(onehot_prec)
            #prec_array = np.array(prec_array)
            full_array.append(prec_array)
        full_array = np.array(full_array)
        if all_precs:
            return all_precs_onehot
        else:
            return full_array
        
# Get the target features as a list from a dataframe
def acquire_target_features(target_df, top_feat=False):
#         top_features = ['MagpieData maximum CovalentRadius',
#          'MagpieData range MendeleevNumber',
#          'MagpieData mean MeltingT',
#          'MagpieData avg_dev NpUnfilled',
#          'MagpieData avg_dev NdValence',
#          'avg ionic char',
#          'MagpieData avg_dev MeltingT',
#          'MagpieData mean GSbandgap',
#          'MagpieData range Column',
#          'MagpieData maximum MendeleevNumber',
#          'MagpieData avg_dev NUnfilled',
#          'MagpieData avg_dev Electronegativity',
#          'MagpieData avg_dev GSvolume_pa',
#          'MagpieData mean NpUnfilled',
#          'MagpieData mean NUnfilled']
    top_features = ['MagpieData maximum CovalentRadius',
     'MagpieData range MendeleevNumber',
     'MagpieData avg_dev NdValence',
     'avg ionic char',
     'MagpieData mean NUnfilled',
     'MagpieData range CovalentRadius',
     'MagpieData mean MeltingT',
     'MagpieData avg_dev MeltingT',
     'MagpieData avg_dev Electronegativity',
     'MagpieData avg_dev NUnfilled',
     'MagpieData mean GSbandgap',
     'MagpieData mean Electronegativity',
     'MagpieData mean NpUnfilled',
     'MagpieData avg_dev CovalentRadius',
     'MagpieData mean CovalentRadius']
    inputs = []
    inputs_full = []
    for index, row in target_df.iterrows():
        if top_feat:
            inputs.append(list(row[top_features].values))
        inputs_full.append(list(row.values)[2:])
    if top_feat:
        return inputs
    else:
        return inputs_full
    
# Append onehot features to target features 
def append_onehot_features(inputs, all_precs_onehot):
    for i in range(len(inputs)):
        inputs[i].extend(list(all_precs_onehot[i]))
        inputs[i] = np.array(inputs[i])
    inputs = np.array(inputs)
    return inputs

# Conduct recursive feature elimination 
def recursive_feature_elimination(inputs, outputs, show_plot = True):
    #X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.20)
    min_features_to_select = 135
    class RandomForestRegressorWithCoef(RandomForestRegressor):
        def fit(self, *args, **kwargs):
            super(RandomForestRegressorWithCoef, self).fit(*args, **kwargs)
            self.coef_ = self.feature_importances_
    rf = RandomForestRegressorWithCoef()
    rfecv = RFECV(estimator=rf, step=5, cv=2, verbose=2, min_features_to_select=min_features_to_select, n_jobs=2)
    selector=rfecv.fit(inputs, outputs)
    print("Number of features selected: {}".format(selector.n_features_))
    rfecv.get_support(indices = True)
    if show_plot:
        start = min_features_to_select
        end = len(rfecv.grid_scores_) * 5 + start
        num = int((end - start) / 5)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(np.linspace(start, end, num, endpoint=False), rfecv.grid_scores_)
        plt.show()
    
    return selector, rfecv

    