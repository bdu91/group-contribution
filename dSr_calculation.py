from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from copy import deepcopy
import numpy as np
from process_thermo_data import thermo_data
from chemaxon import Calculate_mol_properties
thermodynamics_data = thermo_data()

class dSr_calculation(object):
    def __init__(self, dSf_data_frac = 0.0, use_dSr_from_dSf = False):
        """
        A module that calculates dSr of reaction for transformation of dGr over temperature
        """
        # the minimum fraction of dSf data used to calculate dSr in a single reaction
        # if 0 means that in reaction where dSf data are not available for all of its metabolites, dSf estimates are
        # used for all the metabolites to calculate dSr
        self.dSf_data_frac = dSf_data_frac
        # if true, we will use dSr data calculated only from dSf data and predictions
        self.use_dSr_from_dSf = use_dSr_from_dSf
        self.process_dSr_training_data()
        # load the dataframe containing dSf and pKMg data, as well as precalculated molecular descriptors
        self.sid_mol_properties_table = pd.read_csv('data/dSf_pKMg_data.csv')
        self.mol_properties_names = self.sid_mol_properties_table.columns.tolist()[self.sid_mol_properties_table.columns.tolist().index('C_partial_charge'):]
        self.new_sid_properties_dict = {} #a dict that stores the molecular properties calculated
    
    @staticmethod
    def calc_rxn_met_num_change(rxn_dict_list):
        """
        Calculate the difference in number of products and substrates for a list of reactions, not considering water (CHB_15377) in reaction
        :param rxn_dict_list: a list of dictionaries for reactions; in each dictionary, keys are metabolite id, values are stoichiometry of metabolite in reaction
        :return: the differences in number of products and substrates for reactions in a list
        """
        met_num_change_list = []
        rxn_dict_list_cp = deepcopy(rxn_dict_list)
        for cur_rxn_dict in rxn_dict_list_cp:
            if 'CHB_15377' in cur_rxn_dict.keys():
                del cur_rxn_dict['CHB_15377'] #not considering water in the metabolites
            num_of_products = 0
            num_of_reactants = 0
            for stoich in cur_rxn_dict.values():
                if stoich > 0:
                    num_of_products += abs(stoich)
                else:
                    num_of_reactants += abs(stoich)
            cur_met_num_change = num_of_products - num_of_reactants
            met_num_change_list.append(cur_met_num_change)
        return met_num_change_list
    
    def process_dSr_training_data(self):
        """
        Prepare training data for dSr estimation
        :return: self.selected_dSr_training_data_table as pandas dataframe contains information for training data
                 self.dSr_training_data is the training data for dSr estimation
        """
        self.dSr_training_data_table = pd.read_csv('data/dSr_training_data.csv')
        if self.use_dSr_from_dSf == False:
            #dSr calculated from dGr over T or dGr and dHr
            dSr_from_slope_and_dGdH_table = self.dSr_training_data_table[self.dSr_training_data_table['dS_r_from_slope'].notnull()]
            #dSr calculated from dSf data
            dSr_from_dSf_data_table = self.dSr_training_data_table[(self.dSr_training_data_table['dS_r_from_slope'].isnull())\
                                & (self.dSr_training_data_table['extend of using dS_f data'] >= self.dSf_data_frac)]
            self.selected_dSr_training_data_table = dSr_from_slope_and_dGdH_table.append(dSr_from_dSf_data_table, ignore_index = True)
            
            dSr_training_data = []
            for i, row in self.selected_dSr_training_data_table.iterrows():
                if not pd.isnull(row['dS_r_from_slope']):
                    dSr_training_data.append(row['dS_r_from_slope'])
                else:
                    dSr_training_data.append(row['using dS_f data'])
            self.dSr_training_data = np.array(dSr_training_data)
        else:
            self.selected_dSr_training_data_table = self.dSr_training_data_table.copy()
            self.dSr_training_data = np.array(self.selected_dSr_training_data_table['using dS_f data'].tolist())
    
    def _construct_dSr_property_matrix(self, rxn_dicts, rxn_dGr_data):
        """

        :param rxn_dicts: a list of dictionaries for reactions; in each dictionary, keys are metabolite id, values are stoichiometry of metabolite in reaction
        :param rxn_dGr_data: a list of dGr for each reaction
        :return: dSr_property_matrix: the property matrix containing features used for dSr estimation
                 to_subtract_dSr_array: sum of inorganic dSf to subtract for each reaction, we are not considering
                                        inorganic compounds in dSr estimation
        """
        #updated_groups entry is added when new compounds are added to compounds_data_dict
        if 'updated_groups' in self.compounds_data_dict['CHB_15422_-1'].keys(): #random species id to get group info
            groups_to_query = 'updated_groups'
        else:
            groups_to_query = 'groups'
        group_len_to_query = len(self.compounds_data_dict['CHB_15422_-1']['groups'])
        
        to_subtract_dSr_array = np.zeros(len(rxn_dicts))
        to_subtract_dGr_array = np.zeros(len(rxn_dicts))
        dSr_group_matrix = np.zeros((len(rxn_dicts),group_len_to_query))
        mol_properties_matrix = np.zeros((len(rxn_dicts),len(self.mol_properties_names)))

        for i, cur_rxn_dict in enumerate(rxn_dicts):
            for compound_id, stoich in cur_rxn_dict.iteritems():
                cur_sid = self.pH7species_id_dict[compound_id]
                try:
                    #these compounds have dG_f/dS_f data and cannot be broken down into groups
                    to_subtract_dSr_array[i] += self.compounds_data_dict[cur_sid]['dS_f'] * stoich
                    to_subtract_dGr_array[i] += self.compounds_data_dict[cur_sid]['dG_f'] * stoich
                except KeyError:
                    cur_species_group = np.array(self.compounds_data_dict[cur_sid][groups_to_query][:group_len_to_query])
                    dSr_group_matrix[i] += (cur_species_group * stoich)
                    if cur_sid in self.sid_mol_properties_table.species_id.tolist():
                        cur_mol_properties = self.sid_mol_properties_table[self.sid_mol_properties_table['species_id']\
                                             == cur_sid][self.mol_properties_names].values[0]
                    elif cur_sid in self.new_sid_properties_dict.keys():
                        #directly extract properties instead of calculating them
                        cur_mol_properties = self.new_sid_properties_dict[cur_sid]
                    else:
                        cur_mol_properties = Calculate_mol_properties(self.compounds_data_dict[cur_sid]['smiles_form'])
                        self.new_sid_properties_dict[cur_sid] = cur_mol_properties
                        print "Calculated molecular properties for %s" % compound_id
                    mol_properties_matrix[i] += (np.array(cur_mol_properties) * stoich)

        #the features used for dSr estimation, group decompositions, dGr, met_num_change, molecular properties
        rxn_dGr_subtracted = np.array(rxn_dGr_data) - to_subtract_dGr_array
        rxn_met_num_change_array = dSr_calculation.calc_rxn_met_num_change(rxn_dicts)
        dSr_property_matrix = np.append(np.append(dSr_group_matrix, np.transpose([rxn_dGr_subtracted]), axis = 1), np.transpose([rxn_met_num_change_array]), axis = 1)
        dSr_property_matrix = np.hstack((dSr_property_matrix, mol_properties_matrix))

        return dSr_property_matrix, to_subtract_dSr_array
        
    def dSr_predictor(self, rxn2calc_dicts, rxn2calc_dGr_data, compounds_data_dict, pH7species_id_dict):
        """
        Estimator for dSr
        :param rxn2calc_dicts: a list of dictionaries for reactions; in each dictionary, keys are metabolite id, values are stoichiometry of metabolite in reaction
        :param rxn2calc_dGr_data: a list of dGr for each reaction
        :param compounds_data_dict: dictionary that stores information for different protonation states of the compound
        :param pH7species_id_dict: dictionary that maps compound id to its dominant pH 7 species id
        :return: the list of predicted dSr for each reaction
        """
        self.compounds_data_dict = compounds_data_dict
        self.pH7species_id_dict = pH7species_id_dict
        dSr_training_data_rxn_dicts = [thermodynamics_data.parse_formula(cur_formula) for cur_formula in self.selected_dSr_training_data_table['rxn_formula'].tolist()]
        dSr_property_matrix_training, to_subtract_dSr_array_training = self._construct_dSr_property_matrix(dSr_training_data_rxn_dicts,\
                                                                        self.selected_dSr_training_data_table['median_dG_r'])
        dSr_training_data = self.dSr_training_data - to_subtract_dSr_array_training #subtract dSr part due to inorganic compounds

        # remove all 0 columns in property matrix
        dSr_training_mat_0_cols = np.where(~dSr_property_matrix_training.any(axis=0))[0]
        dSr_property_matrix_training = np.delete(dSr_property_matrix_training, dSr_training_mat_0_cols, 1)

        # now set up the lasso regression model
        best_lasso_a = 0.00123284673944
        # make sure standardization is applied before training and prediction
        dSr_rg = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=best_lasso_a, max_iter=500000, tol=0.001))
        dSr_rg.fit(dSr_property_matrix_training, dSr_training_data)

        #now make the prediction
        dSr_property_matrix_predict, to_subtract_dSr_array_predict = self._construct_dSr_property_matrix(rxn2calc_dicts, rxn2calc_dGr_data)
        #remove columns according the index in training matrix
        dSr_property_matrix_predict = np.delete(dSr_property_matrix_predict, dSr_training_mat_0_cols, 1)
        dSr_prediction = dSr_rg.predict(dSr_property_matrix_predict)
        dSr_prediction = dSr_prediction + to_subtract_dSr_array_predict #add it back to get the actual dSr
        
        return dSr_prediction
    
    def get_TECRDB_rxn_dSr_dict(self, TECRDB_Keq_data_dict):
        """
        Calculate dSr for reactions in TECRDB
        :param TECRDB_Keq_data_dict: generated from module process_thermo_data, the dictionary that stores thermodynamic
               measurements of reactions key being reaction id, value being the info of reaction measurement
        :return: a dictionary with key being reaction id, value being dSr of the reaction
        """
        TECRDB_rxn_dSr_dict = {}
        rxn_formula_to_dSr_dict = dict(zip(self.selected_dSr_training_data_table['rxn_formula'].tolist(), list(self.dSr_training_data)))
        for rxn_id in TECRDB_Keq_data_dict.keys():
            cur_rxn_formula = TECRDB_Keq_data_dict[rxn_id]['rxn_formula']
            TECRDB_rxn_dSr_dict[rxn_id] = rxn_formula_to_dSr_dict[cur_rxn_formula]
        return TECRDB_rxn_dSr_dict