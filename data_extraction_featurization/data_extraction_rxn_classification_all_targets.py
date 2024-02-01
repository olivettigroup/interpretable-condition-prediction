import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from statistics import stdev
import json
import re
from exclude import *
from copy import deepcopy

# Extracts data from the json database and saves it in a more parseable format
def extract_solidstate(reactions, min_pre=2, max_pre=10):
    ss_precursor_nomenclature = dict()
    with open('/home/jupyter/CJK/TempTime/data/sg_identified.json') as f:
        sg_only = json.load(f)
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    extracted_data = []
    bad_list = ['*', '-', 'x', '+', '/', 'ac', '(2N)', '(3N)', '(4N)', '(5N)', '(6N)', '7LiOH', '2Ni(OH)2']
    for rxn in reactions:
        # check if we have bad doi
        found_bad = False
        if rxn["doi"] in BAD_DOI:
            continue
        # check if number of precursors is within limits
        # check if we are extracting one number or not
        # check if we have bad precursor or target
#         if len(rxn["precursors"]) < min_pre or len(rxn["precursors"]) > max_pre:
#             continue
        if len(rxn["precursors"]) < min_pre:
            continue
        #check if any precursor is a pure element, likely obsolete due to project scope
#         if any(rxn["precursors"][x]["material_formula"] in all_elements for x in range(len(rxn["precursors"]))):
#             continue
        if any(rxn["targets_string"][x] in BAD_TARGETS for x in range(len(rxn["targets_string"]))):
            continue
        if any(rxn["precursors"][x]["material_formula"] in BAD_PRECURSORS for x in range(len(rxn["precursors"]))):
            continue
        for bad in bad_list:
            if any(bad in rxn["targets_string"][x] for x in range(len(rxn["targets_string"]))) or any(
                    bad in rxn["precursors"][x]["material_formula"] for x in range(len(rxn["precursors"]))):
                found_bad = True
        if found_bad:
            continue
            
        operation_order, precursor_list = get_synthesis_context(rxn['operations'], rxn['precursors'])
        
        orig_target = get_target_information(rxn['targets_string'])
        targets = list(set(get_all_targets(rxn['targets_string'])))
        operation_times_temps = get_time_temp_conditions(rxn['operations'])

        bad = False
        for sg_rxn in sg_only:
            if sg_rxn['target'] == orig_target and set(sg_rxn['precursors']) == set(precursor_list):
                bad = True
                break
        if bad:
            continue
        
        # manual correction
        if rxn["doi"] == '10.1016/j.jssc.2012.01.053':
            orig_target = "Li0.02Zn0.2Ga0.02O0.24"
        precursor_list = remove_hydrates(precursor_list)
        if any('.' in prec for prec in precursor_list):
            continue
        precursor_list = replace_precursors(precursor_list)
        if any(prec in BAD_PRECURSORS for prec in precursor_list):
            continue
            
            
        # extract all targets separately
        for tar in targets:
            # save most commonly reported name for precursor composition
            for precursor in precursor_list:
                comp = Composition(Composition(precursor).reduced_formula)
                if comp not in ss_precursor_nomenclature:
                    ss_precursor_nomenclature[comp] = []
                ss_precursor_nomenclature[comp].append(precursor)
            # check that precursors share at least one element with target
            precursor_list_modified = check_precursors(deepcopy(precursor_list), tar)
            if len(precursor_list_modified) > max_pre or len(precursor_list_modified) < min_pre:
                continue

            extracted_data.append({
                # lower DOI to standardize them
                'DOI': rxn["doi"].lower(),
                'target': tar,
                'precursors': precursor_list_modified,
                'operation_types': [x[0] for x in operation_order],
                'operation_tokens': [x[1].lower() for x in operation_order],
                'operation_times': [x[0] for x in operation_times_temps],
                'operation_temps': [x[1] for x in operation_times_temps]
            }
            )
            
    print("Returning extracted data of {}/{} reactions.".format(len(extracted_data), len(reactions)))
    # save most common
    for key in ss_precursor_nomenclature:
        ss_precursor_nomenclature[key] = max(ss_precursor_nomenclature[key], key=ss_precursor_nomenclature[key].count)
    return extracted_data, ss_precursor_nomenclature

def extract_solgel(reactions, min_pre=2, max_pre=10):
    sg_precursor_nomenclature = dict()
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    with open('/home/jupyter/CJK/TempTime/data/ss_identified.json') as f:
        ss_only = json.load(f)
    extracted_data = []
    for rxn in reactions:
#         if len(rxn["precursors"]) < min_pre or len(rxn["precursors"]) > max_pre:
#             continue
        if len(rxn["precursors"]) < min_pre:
            continue
        if any(rxn["targets_string"][x] in BAD_TARGETS_SOLGEL for x in range(len(rxn["targets_string"]))):
            continue
        #check if any precursor is a pure element
        if any(rxn["precursors"][x]["material_formula"] in all_elements for x in range(len(rxn["precursors"]))):
            continue
        if any(rxn["precursors"][x]["material_formula"] in BAD_PRECURSORS_SOLGEL for x in range(len(rxn["precursors"]))):
            continue
        operation_order, precursor_list = get_synthesis_context(rxn['operations'], rxn['precursors'])
        orig_target = get_target_information(rxn['targets_string'])
        targets = list(set(get_all_targets(rxn['targets_string'])))
        operation_times_temps = get_time_temp_conditions(rxn['operations'])
        
        bad = False
        for ss_rxn in ss_only:
            if ss_rxn['target'] == orig_target and set(ss_rxn['precursors']) == set(precursor_list):
                bad = True
                break
        if bad:
            continue
            
        precursor_list = remove_hydrates(precursor_list)
        if any('.' in prec for prec in precursor_list):
            continue
        precursor_list = replace_precursors(precursor_list)
        if any(prec in BAD_PRECURSORS_SOLGEL for prec in precursor_list):
            continue
            
        # extract all targets separately
        for tar in targets:
            # save most commonly reported name for precursor composition
            for precursor in precursor_list:
                comp = Composition(Composition(precursor).reduced_formula)
                if comp not in sg_precursor_nomenclature:
                    sg_precursor_nomenclature[comp] = []
                sg_precursor_nomenclature[comp].append(precursor)
            # check that precursors share at least one element with target
            precursor_list_modified = check_precursors(deepcopy(precursor_list), tar)
            if len(precursor_list_modified) > max_pre or len(precursor_list_modified) < min_pre:
                continue
            extracted_data.append({
                # lower DOI to standardize them
                'DOI': rxn["doi"].lower(),
                'target': tar,
                'precursors': precursor_list_modified,            
                'operation_types': [x[0] for x in operation_order],
                'operation_tokens': [x[1].lower() for x in operation_order],
                'operation_times': [x[0] for x in operation_times_temps],
                'operation_temps': [x[1] for x in operation_times_temps]
            }
        )
        

    print("Returning extracted data of {}/{} reactions.".format(len(extracted_data), len(reactions)))
    # save most common
    for key in sg_precursor_nomenclature:
        sg_precursor_nomenclature[key] = max(sg_precursor_nomenclature[key], key=sg_precursor_nomenclature[key].count)
    return extracted_data, sg_precursor_nomenclature

def extract_solution(reactions, min_pre=2, max_pre=10):
    sol_precursor_nomenclature = dict()
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    with open('/home/jupyter/CJK/TempTime/data/ss_identified.json') as f:
        ss_only = json.load(f)
    extracted_data = []
    for rxn in reactions:
#         if len(rxn["precursors"]) < min_pre or len(rxn["precursors"]) > max_pre:
#             continue
        if len(rxn["precursors"]) < min_pre:
            continue
        if any(rxn["targets_string"][x] in BAD_TARGETS_SOLGEL for x in range(len(rxn["targets_string"]))):
            continue
        #check if any precursor is a pure element
        if any(rxn["precursors"][x]["material_formula"] in all_elements for x in range(len(rxn["precursors"]))):
            continue
        if any(rxn["precursors"][x]["material_formula"] in BAD_PRECURSORS_SOLGEL for x in range(len(rxn["precursors"]))):
            continue
        operation_order, precursor_list = get_synthesis_context(rxn['operations'], rxn['precursors'], solution=True)
        if rxn['targets_string'] is None:
            continue
        orig_target = get_target_information([rxn['targets_string']])
        targets = get_all_targets([rxn['targets_string']])
        # if we get no good targets from this recipe
        if targets is None:
            continue
        targets = list(set(targets))
        operation_times_temps = get_time_temp_conditions_sol(rxn['operations'])
        
        bad = False
        for ss_rxn in ss_only:
            if ss_rxn['target'] == orig_target and set(ss_rxn['precursors']) == set(precursor_list):
                bad = True
                break
        if bad:
            continue
            
        precursor_list = remove_hydrates(precursor_list)
        if any('.' in prec for prec in precursor_list):
            continue
        precursor_list = replace_precursors(precursor_list)
        if any(prec in BAD_PRECURSORS_SOLGEL for prec in precursor_list):
            continue
            
        # extract all targets separately
        for tar in targets:
            if tar in ['PdWCSNHs','CfB6H5O13','CuPdWCSNHs','NHs']:
                continue
            # save most commonly reported name for precursor composition
            for precursor in precursor_list:
                try:
                    comp = Composition(Composition(precursor).reduced_formula)
                    if comp not in sol_precursor_nomenclature:
                        sol_precursor_nomenclature[comp] = []
                    sol_precursor_nomenclature[comp].append(precursor)
                except:
                    pass
            # check that precursors share at least one element with target
            precursor_list_modified = check_precursors(deepcopy(precursor_list), tar)
            if len(precursor_list_modified) > max_pre or len(precursor_list_modified) < min_pre:
                continue
            extracted_data.append({
                # lower DOI to standardize them
                'DOI': rxn["doi"].lower(),
                'target': tar,
                'precursors': precursor_list_modified,            
                'operation_types': [x[0] for x in operation_order],
                'operation_tokens': [x[1].lower() for x in operation_order],
                'operation_times': [x[0] for x in operation_times_temps],
                'operation_temps': [x[1] for x in operation_times_temps],
                'type': rxn['type']
            }
        )
        

    print("Returning extracted data of {}/{} reactions.".format(len(extracted_data), len(reactions)))
    # save most common
    for key in sol_precursor_nomenclature:
        sol_precursor_nomenclature[key] = max(sol_precursor_nomenclature[key], key=sol_precursor_nomenclature[key].count)
    return extracted_data, sol_precursor_nomenclature

def get_synthesis_context(operations: list, precursors: list, solution=False):
    if solution:
        operation_order = [(op['type'], op['string']) for op in operations]
    else:
        operation_order = [(op['type'], op['token']) for op in operations]
    precursors_list = [pre["material_formula"] for pre in precursors]
    # check if we have hydrates and truncate formula
    for i, pre in enumerate(precursors_list):
        splitted = pre.split('·')
        if len(splitted) > 1:
            precursors_list[i] = splitted[0]
    # check if we have acetates and convert to proper chemical formula, and other errata
    for i in range(len(precursors_list)):
        if 'Ac' in precursors_list[i]:
            precursors_list[i] = precursors_list[i].replace('Ac', 'CH3COO')
        elif precursors_list[i] == 'Co2':
            precursors_list[i] = 'CO2'
        elif precursors_list[i] == 'MnO2(97.5)':
            precursors_list[i] = 'MnO2'
        elif precursors_list[i] == 'Sm3':
            precursors_list[i] = 'Sm'
        elif precursors_list[i] == 'Dy3':
            precursors_list[i] = 'Dy'
        elif precursors_list[i] == 'Am0.07':
            precursors_list[i] = 'Am'
        elif precursors_list[i] == '(MgCO3)4Mg(OH)2.5H2O':
            precursors_list[i] = '(MgCO3)4Mg(OH)2'
    if '(Zr,Sn,Ti)O3' in precursors_list:
            precursors_list.remove('(Zr,Sn,Ti)O3')
        
    if precursors_list == ['Bi(N03)3', 'Fe(N03)3', 'Ba(N03)2']:
        precursors_list = ['Bi(NO3)3', 'Fe(NO3)3', 'Ba(NO3)2']
    return operation_order, precursors_list

# Get the target from the list of targets reported in the synthesis
def get_target_information(targets):
    # take average of tried compositions for target if > 1
    if len(targets) > 1:
        target = targets[round(len(targets) / 2)]
    else:
        target = targets[0]
    return target


# Remove duplicate entries frothe database (share same target and precursors)
def remove_duplicates(db, solution=False, target_only=False, precursor_only=False):
    initial_length = len(db)
    all_targets_precs = []
    for rxn in db:
        found = False
        for item in all_targets_precs:
            if (not target_only and not precursor_only and rxn["target"] == item["target"] and set(rxn["precursors"]) == set(item["precursors"])) or (target_only and rxn["target"] == item["target"]) or (precursor_only and set(rxn["precursors"]) == set(item["precursors"])):
                item["DOIs"].append(rxn["DOI"])
                item["count"] += 1
                found = True
                if solution:
                    item["type"].append(rxn["type"])
        if not found:
            if solution:
                if not target_only and not precursor_only:
                    all_targets_precs.append({
                        "target": rxn["target"],
                        "precursors": rxn["precursors"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1,
                        "type": [rxn["type"]]
                    })
                elif precursor_only:
                    all_targets_precs.append({
                        "precursors": rxn["precursors"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1,
                        "type": [rxn["type"]]
                    })
                else:
                    all_targets_precs.append({
                        "target": rxn["target"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1,
                        "type": [rxn["type"]]
                    })
            else:
                if not target_only and not precursor_only:
                    all_targets_precs.append({
                        "target": rxn["target"],
                        "precursors": rxn["precursors"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1
                    })
                elif precursor_only:
                    all_targets_precs.append({
                        "precursors": rxn["precursors"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1
                    })
                else:
                    all_targets_precs.append({
                        "target": rxn["target"],
                        "DOIs": [rxn["DOI"]],
                        "count": 1
                    })  
    print("After removing duplicates, remaining extracted data contains {}/{} reactions.".format(len(all_targets_precs), initial_length))
    return all_targets_precs

# Acquire temperature and time conditions from the database operations
def get_time_temp_conditions(operations: list, average=True, min_temp=0, max_temp=2000, min_time=0, max_time=100):
    operation_times_temps = []
    for i, op in enumerate(operations):
        time = op['conditions']['heating_time']
        time_values = None

        temp = op['conditions']['heating_temperature']
        temp_values = None

        if time:
            if time[0]["units"] in ["min", "minutes"]:
                time_values = [val / 60.0 for val in time[0]["values"]]
            elif time[0]["units"] in ["day", "d"]:
                time_values = [val * 24.0 for val in time[0]["values"]]
            else:
                time_values = time[0]["values"]
            if not time_values:
                time_values = None
            elif average:
                time_values = round(np.average(time_values))
                if time_values <= min_time or time_values > max_time:
                    time_values = None
                

        if temp:
            if len(temp) > 1:
                ind = np.argmax([x["max_value"] for x in temp])
                temp_vec = temp[ind]
            else:
                temp_vec = temp[0]
                
            if temp_vec["units"] == "K":
                temp_values = [val - 273.0 for val in temp_vec['values']]
            else:
                temp_values = temp_vec['values']
            if not temp_values:
                temp_values = None
            elif average:
                #temp_values = round(np.average(temp_values))
                # modifying to take max value
                max_temp_ind = np.argmax(temp_values)
                temp_values = temp_values[max_temp_ind]
                if temp_values <= min_temp or temp_values > max_temp:
                    temp_values = None

                

        operation_times_temps.append((time_values, temp_values))
    return operation_times_temps


    print("Returning extracted data of {}/{} reactions.".format(len(extracted_data), len(reactions)))
    return extracted_data

def get_time_temp_conditions_sol(operations: list, average=True, min_temp=0, max_temp=2000, min_time=0, max_time=100):
    operation_times_temps = []
    for i, op in enumerate(operations):
        time = op['conditions']['time']
        time_values = None

        temp = op['conditions']['temperature']
        temp_values = None

        if time:
            if time["units"] in ["min", "minutes"]:
                time_values = [val / 60.0 for val in time["values"]]
            elif time["units"] in ["day", "d"]:
                time_values = [val * 24.0 for val in time["values"]]
            else:
                time_values = time["values"]
            if not time_values:
                time_values = None
            elif average:
                time_values = round(np.average(time_values))
                if time_values <= min_time or time_values > max_time:
                    time_values = None
                

        if temp:
            temp_vec = temp
                
            if temp_vec["units"] == "K":
                temp_values = [val - 273.0 for val in temp_vec['values']]
            else:
                temp_values = temp_vec['values']
            if not temp_values:
                temp_values = None
            elif average:
                #temp_values = round(np.average(temp_values))
                # modifying to take max value
                max_temp_ind = np.argmax(temp_values)
                temp_values = temp_values[max_temp_ind]
                if temp_values <= min_temp or temp_values > max_temp:
                    temp_values = None

                

        operation_times_temps.append((time_values, temp_values))
    return operation_times_temps


    print("Returning extracted data of {}/{} reactions.".format(len(extracted_data), len(reactions)))
    return extracted_data

def remove_hydrates(material):
    # remove hydrates
    for i in range(len(material)):
        if material[i] != 'H2O' and 'H2O' in material[i]:
            material[i] = material[i].replace('.H2O', '').replace('.2H2O', '').replace('.3H2O', '').replace('.4H2O', '').replace('.5H2O', '').replace('.6H2O', '').replace('.7H2O', '').replace('.8H2O', '').replace('.9H2O', '').replace('.xH2O', '').replace('nH2O', '').replace('xH2O', '').replace('x2H2O', '').replace('x3H2O', '').replace('x4H2O', '').replace('x5H2O', '').replace('x6H2O', '').replace('x7H2O', '').replace('x8H2O', '').replace('x9H2O', '').replace('-H2O', '').replace('-2H2O', '').replace('-3H2O', '').replace('-4H2O', '').replace('-5H2O', '').replace('-6H2O', '').replace('-7H2O', '').replace('-8H2O', '').replace('-9H2O', '').replace(',H2O', '').replace(',2H2O', '').replace(',3H2O', '').replace(',4H2O', '').replace(',5H2O', '').replace(',6H2O', '').replace(',7H2O', '').replace(',8H2O', '').replace(',9H2O', '').replace('.(H2O)', '').replace('.2(H2O)', '').replace('.3(H2O)', '').replace('.4(H2O)', '').replace('.5(H2O)', '').replace('.6(H2O)', '').replace('.7(H2O)', '').replace('.8(H2O)', '').replace('.9(H2O)', '').replace('*H2O', '').replace('*2H2O', '').replace('*3H2O', '').replace('*4H2O', '').replace('*5H2O', '').replace('*6H2O', '').replace('*7H2O', '').replace('*8H2O', '').replace('*9H2O', '').replace('(2H2O)', '').replace('(3H2O)', '').replace('(4H2O)', '').replace('(5H2O)', '').replace('(6H2O)', '').replace('(7H2O)', '').replace('(8H2O)', '').replace('(9H2O)', '')
            material[i] = material[i].replace('(H2O)2', '').replace('(H2O)3', '').replace('(H2O)4', '').replace('(H2O)5', '').replace('(H2O)6', '').replace('(H2O)7', '').replace('(H2O)8', '').replace('(H2O)9', '').replace('(H2O)', '')
            # remove ones with numbers first
            for j in range(2, 10):
                hydrate = str(j) + r'H2O$'
                material[i] = re.sub(hydrate, '', material[i])
            hydrate_2 = r'H2O$'
            material[i] = re.sub(hydrate_2, '', material[i])
    return material
def replace_precursors(material):
    # replace faulty materials
    for key in PREC_REPLACEMENTS:
        for i in range(len(material)):
            if material[i] == key:
                material[i] = PREC_REPLACEMENTS[key]
    if "" in material:
        material.remove("")
    return material

def check_precursors(material_list, target):
    target_comp = Composition(target)
    target_elements = [Element(e).symbol for e in target_comp.elements if e.symbol not in ["O", "H"]]
    good_precursors = []
    for material in material_list:
        try:
            precursor_comp = Composition(material)
            precursor_elems = [Element(e).symbol for e in precursor_comp.elements if e.symbol not in ["O", "H"]]
        except:
            continue
        precursor_elems = [Element(e).symbol for e in precursor_comp.elements if e.symbol not in ["O", "H"]]
        if any(elem in target_elements for elem in precursor_elems):
            # return standardized precursor
            good_precursors.append(precursor_comp.reduced_formula)
    # return sorted precursors
    return sorted(good_precursors)

def get_all_targets(targets):
    good_targets = []
    for tar in targets:
        if tar == "":
            continue
        try:
            comp = Composition(tar)
            target_elements = [Element(e).symbol for e in comp.elements if e.symbol not in ["O", "H"]]
        except:
            continue
        good_targets.append(tar)
    # return one target per recipe, standardized, for classification task
    if len(good_targets) == 0:
        return None
#     mid_target = good_targets[round(len(good_targets) / 2)]
#     mid_target = Composition(mid_target).reduced_formula
    return good_targets

