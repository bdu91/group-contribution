from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from process_thermo_data import thermo_data
from chemaxon import Calculate_mol_properties
import compound_groups

class dSr_calculation(object):
    def __init__(self, compounds_data_dict = {}, compounds_pH7_species_id_dict = {}):
        """
        A module that calculates dSr of reaction for transformation of dGr over temperature
        """
        # the minimum fraction of dSf data used to calculate dSr in a single reaction
        # if 0 means that in reaction where dSf data are not available for all of its metabolites, dSf estimates are
        self.dSf_pKMg_data_df = pd.read_csv('data/dSf_pKMg_data.csv')
        self.dSr_data_df = pd.read_csv('data/dSr_training_data.csv')
        self.mol_properties_names = self.dSf_pKMg_data_df.columns.tolist()[self.dSf_pKMg_data_df.columns.tolist().index('C_partial_charge'):]
        self.process_dSf_training_data()
        self.compounds_pH7_species_id_dict = compounds_pH7_species_id_dict
        self.compounds_data_dict = compounds_data_dict
        self.cofactor_cids = ['CHB_16048', 'CHB_17621', 'PBC_4481', 'PBC_4066204', 'CHB_16238', 'CHB_17877', 'CHB_15846', 'CHB_16908', 'CHB_18009', 'CHB_16474', 'PBC_123926', 'MAN_10059', 'PBC_4156341', 'MAN_10151']
        self.new_sid_properties_dict = {} #a dict that stores the molecular properties calculated
    
    def process_dSf_training_data(self):
        """
        Process the training data for dSf calculation, including data for dSf and dSr (linear combination of dSr)
        """
        self.dSf_training_sids = self.dSf_pKMg_data_df[~pd.isnull(self.dSf_pKMg_data_df['dS_f(J/K/mol)'])].species_id.tolist()
        self.dSr_training_formula_list = self.dSr_data_df.rxn_formula.tolist()
        self.dSf_data = [self.dSf_pKMg_data_df[self.dSf_pKMg_data_df.species_id == cur_sid]['dS_f(J/K/mol)'].tolist()[0] for cur_sid in self.dSf_training_sids]
        self.dSr_data = [self.dSr_data_df[self.dSr_data_df.rxn_formula == cur_formula].dS_r_from_slope.tolist()[0] for cur_formula in self.dSr_training_formula_list]
        self.dSf_data_total = np.array(self.dSf_data + self.dSr_data)
        
    def get_sids_from_formula(self, rxn_formula_list):
        """
        Obtain all the species ids from the list of reaction formulas
        :param rxn_formula_list: a list of reaction formulas, each reaction formula can either be a string such as "2 CHB_16761 = CHB_15422 + CHB_16027" or a dictionary such as {'CHB_15422': 1, 'CHB_16027': 1, 'CHB_16761': -2.0}
        :return: the list of species ids at pH7 in the reaction formulas
        """
        cid_list_from_formula = []
        for cur_formula in rxn_formula_list:
            if type(cur_formula) == dict:
                cid_list_from_formula += cur_formula.keys()
            else:
                cid_list_from_formula += thermo_data.parse_formula(cur_formula).keys()
        cid_list_from_formula = list(set(cid_list_from_formula))
        sid_list_from_formula = [self.compounds_pH7_species_id_dict[cur_cid] for cur_cid in cid_list_from_formula]
        return sid_list_from_formula
    
    def build_property_mat(self, sid_list2query = [], rxn_formula_list = []):
        """
        Construct the property matrix to estimate dSf
        :param sid_list2query: a list of species ids, the aqueous species to get the groups and molecular properties
        :param rxn_formula_list: a list of reaction formula or reaction dictionaries, the reaction to get the sum of groups and molecular properties from its participating aqueous species
        :return: numpy array, the property matrix with row corresponding to aqueous species and reactions, columns are groups and molecular properties
        """
        #for dSf training data, we also have dSr calculated from dSf, that is effectively the stoichiometric sum of multiple dSf
        sid_list = list(set(sid_list2query + self.get_sids_from_formula(rxn_formula_list)))

        sid2get_groups = \
        list(set(sid_list) - set([sid for sid in self.compounds_data_dict.keys() if 'groups' in self.compounds_data_dict[sid].keys()])) + \
        list(set.intersection(set(sid_list),set([self.compounds_pH7_species_id_dict[cid] for cid in self.cofactor_cids])))
        sid2get_groups_smiles_forms = []
        for sid in sid2get_groups:
            if sid in self.compounds_data_dict.keys():
                sid2get_groups_smiles_forms.append(self.compounds_data_dict[sid]['smiles_form'])
            else:
                sid2get_groups_smiles_forms.append(self.dSf_pKMg_data_df[self.dSf_pKMg_data_df['species_id']==sid].smiles_form.tolist()[0])
        sid2get_groups_gmat = compound_groups.get_group_matrix(sid2get_groups_smiles_forms)
        group_num = compound_groups.get_group_matrix(['CC']).shape[1] #defined groups that are meaningful, index over this range means the compound is nondecomposable

        #get dSf group matrix and dSf_property matrix
        dSf_group_mat = np.zeros((len(sid_list2query) + len(rxn_formula_list), group_num))
        property_startInd = self.dSf_pKMg_data_df.columns.tolist().index('C_partial_charge')
        property_endInd = self.dSf_pKMg_data_df.columns.tolist().index('TPSA')
        dSf_property_mat = np.zeros((len(sid_list2query) + len(rxn_formula_list), property_endInd + 1 - property_startInd))
        to_subtract_dSr_array = np.zeros(len(sid_list2query) + len(rxn_formula_list)) #to substract off inorganic compounds
        for i, cur_entry in enumerate(sid_list2query + rxn_formula_list):
            if i < len(sid_list2query):
                #is sid
                if cur_entry in sid2get_groups:
                    dSf_group_mat[i] = sid2get_groups_gmat[sid2get_groups.index(cur_entry)][:group_num]
                else:
                    dSf_group_mat[i] = np.array(self.compounds_data_dict[cur_entry]['groups'][:group_num])
                dSf_property_mat[i] = self.dSf_pKMg_data_df[self.dSf_pKMg_data_df.species_id == cur_entry].iloc[:,property_startInd:property_endInd+1].values[0]
            else:
                #is reaction
                cur_rxn_dict = cur_entry if type(cur_entry) == dict else thermo_data.parse_formula(cur_entry)
                for cur_cid, stoich in cur_rxn_dict.iteritems():
                    cur_sid = self.compounds_pH7_species_id_dict[cur_cid]
                    #construct group matrix for the reaction
                    if cur_sid in sid2get_groups:
                        #nondecomposable groups will effectively have 0 contribution to groups
                        dSf_group_mat[i] += stoich * sid2get_groups_gmat[sid2get_groups.index(cur_sid)][:group_num]
                    else:
                        dSf_group_mat[i] += stoich * np.array(self.compounds_data_dict[cur_sid]['groups'][:group_num])
                    #construct property matrix
                    try:
                        #inorganic compounds are not considered for dSr fitting
                        to_subtract_dSr_array[i] += self.compounds_data_dict[cur_sid]['dS_f'] * stoich
                    except KeyError:
                        if cur_sid in self.dSf_pKMg_data_df.species_id.tolist():
                            cur_mol_properties = self.dSf_pKMg_data_df[self.dSf_pKMg_data_df.species_id == cur_sid].iloc[:,property_startInd:property_endInd+1].values[0]
                        elif cur_sid in self.new_sid_properties_dict.keys():
                            cur_mol_properties = self.new_sid_properties_dict[cur_sid]
                        else:
                            print "Calculating molecular properties for %s" % cur_cid
                            cur_mol_properties = Calculate_mol_properties(self.compounds_data_dict[cur_sid]['smiles_form'])
                            self.new_sid_properties_dict[cur_sid] = cur_mol_properties
                        dSf_property_mat[i] += stoich * np.array(cur_mol_properties)

        #print dSf_group_mat.shape
        #print dSf_property_mat.shape
        dSf_group_property_mat = np.hstack((dSf_group_mat, dSf_property_mat))
        return dSf_group_property_mat, to_subtract_dSr_array
    
    def _train(self):
        """
        Train the regression model for dSf estimation, using both dSf and dSr data as training data
        Standardization is performed before fitting
        """
        self.dSr_training_rdict_list = [thermo_data.parse_formula(cur_formula) for cur_formula in self.dSr_training_formula_list]
        dSf_property_mat_train, to_subtract_dSr_array_train = self.build_property_mat(self.dSf_training_sids, self.dSr_training_rdict_list)
        self.dSf_train_mat_0_cols = np.where(~dSf_property_mat_train.any(axis=0))[0]
        self.dSf_property_mat_train = np.delete(dSf_property_mat_train, self.dSf_train_mat_0_cols, 1)
        self.dSf_train_data = self.dSf_data_total - to_subtract_dSr_array_train
        
        best_lasso_alpha = 0.000225701971963 #selected l1 alpha for the final model
        self.dSf_rg = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=best_lasso_alpha, max_iter=250000, tol=0.001))
        self.dSf_rg.fit(self.dSf_property_mat_train, self.dSf_train_data/1000)
        
    def dSr_predictor(self, rxn2calc_dicts, compounds_data_dict, compounds_pH7_species_id_dict, sid2predict_dSf = []):
        """
        Predict dSr or dSf for given reactions or aqueous species
        :param rxn2calc_dicts: a list of reactions to estimated dSr; each reaction can either be a string such as "2 CHB_16761 = CHB_15422 + CHB_16027" or a dictionary such as {'CHB_15422': 1, 'CHB_16027': 1, 'CHB_16761': -2.0}
        :param compounds_data_dict: dictionary that stores information for different protonation states of the compound
        :param pH7species_id_dict: dictionary that maps compound id to its dominant pH 7 species id
        :param sid2predict_dSf: the list of aqueous species ids to predict dSf
        :return: the list of predicted dSr for reactions (and dSf for aqueous species)
        """
        self.compounds_data_dict = compounds_data_dict
        self.compounds_pH7_species_id_dict = compounds_pH7_species_id_dict
        self._train()
        rxn2calc_dicts = [cur_rxn if type(cur_rxn) == dict else thermo_data.parse_formula(cur_rxn) for cur_rxn in rxn2calc_dicts]
        dSf_property_mat_predict, to_subtract_dSr_array_predict = self.build_property_mat(sid_list2query = sid2predict_dSf, rxn_formula_list = rxn2calc_dicts)
        dSf_prediction = self.dSf_rg.predict(np.delete(dSf_property_mat_predict, self.dSf_train_mat_0_cols, 1))
        dSf_prediction = dSf_prediction*1000 + to_subtract_dSr_array_predict
        
        for i, cur_rxn_dict in enumerate(rxn2calc_dicts):
            if cur_rxn_dict in self.dSr_training_rdict_list:
                dSf_prediction[i+len(sid2predict_dSf)] = self.dSr_data[self.dSr_training_rdict_list.index(cur_rxn_dict)]
        
        return dSf_prediction
        
    def get_TECRDB_rxn_dSr_dict(self, TECRDB_Keq_data_dict, compounds_data_dict, compounds_pH7_species_id_dict):
        """
        Calculate dSr for all reactions in TECRDB
        :param TECRDB_Keq_data_dict: generated from module process_thermo_data, the dictionary that stores thermodynamic measurements of reactions key being reaction id, value being the info of reaction measurement
        :param compounds_data_dict: dictionary that stores information for different protonation states of the compound
        :param pH7species_id_dict: dictionary that maps compound id to its dominant pH 7 species id
        :return: a dictionary with key being reaction id, value being dSr of the reaction
        """
        #assign 0 to reactions whose compounds do not have a smiles form or do not have properties calculated, these reactions all have measured data at 298.15 K, so will not affect the outcome of temperature corrections
        pseudo_formula_to_dSr_dict = {'CHB_17908 = CHB_17513': 0, 'CHB_16389 = CHB_17976': 0, 'CHB_16374 = CHB_18151': 0, 'CHB_18191 = CHB_15033': 0, 'CHB_28262 = CHB_17437': 0, 'CHB_15724 = CHB_18139': 0, 'CHB_28192 = CHB_55437': 0, 'MNXM_145809 = MNXM_6271': 0}
        TECRDB_rxn_formula = list(set([cur_rxn_dict['rxn_formula'] for cur_rxn_dict in TECRDB_Keq_data_dict.values() if cur_rxn_dict['rxn_formula'] not in pseudo_formula_to_dSr_dict.keys()]))
        TECRDB_rxn_dSr_dict = {}
        rxn_formula_dSr_dict = dict(zip(TECRDB_rxn_formula, self.dSr_predictor(TECRDB_rxn_formula, compounds_data_dict, compounds_pH7_species_id_dict)))
        rxn_formula_dSr_dict.update(pseudo_formula_to_dSr_dict)
        for rxn_id in TECRDB_Keq_data_dict.keys():
            cur_rxn_formula = TECRDB_Keq_data_dict[rxn_id]['rxn_formula']
            TECRDB_rxn_dSr_dict[rxn_id] = rxn_formula_dSr_dict[cur_rxn_formula]
        return TECRDB_rxn_dSr_dict