import compound_groups
import numpy as np
from training_data import training_data
from thermodynamic_transformations import thermodynamic_transformations
from dSr_calculation import dSr_calculation
from process_thermo_data import thermo_data
from chemaxon import Get_pKas_Hs_zs_pH7smiles
import sys

class group_contribution(training_data):
    
    def __init__(self, metal_correction=False, T_correction=True):
        """
        A module that predicts dGr/dGf using group contribution, inherit from training_data
        :param metal_correction: whether to perform correction on metal concentration
        :param T_correction: whether to perform correction on temperature
        """
        training_data.__init__(self, metal_correction, T_correction)
        self.get_training_data_and_matrix()
        # group definitions include individual groups + nondecompose_sids + inorg_cpd_with_thermo_data + cofactor_cids
        self.group_def_num = len(self.TECRDB_compounds_data_dict['CHB_15422_-1']['groups']) #subject to change with added nondecomposable compounds
        self.nondecompose_sids = ['CHB_16517_-3','CHB_18036_-3','CHB_15429_0','CHB_16215_-2', 'PBC_5289158_-3', 'CHB_17908_1', 'CHB_17513_0', 'CHB_28262_0', 'CHB_28192_0', 'CHB_55437_0', 'MNXM_145809_0', 'MNXM_6271_0']
        # excluding compounds that are treated as a single group
        self.group_only_num = self.group_def_num - len(self.nondecompose_sids + self.inorg_cpd_with_thermo_data + self.cofactor_cids)
        self.TECRDB_compound_ids = list(set([species_info_dict['compound_id'] for species_info_dict in self.TECRDB_compounds_data_dict.values()]))
        self.dSr_calc = dSr_calculation()
    
    def formula2dict(self, rxn_formula_list):
        """
        Convert a list of reaction formula to a list of dictionary, each with key is metabolite, value is stoichiometry
        """
        return [thermo_data.parse_formula(cur_formula) for cur_formula in rxn_formula_list]


    def _update_cpd_data_dict(self):
        """
        Check if the compound id of the provided reaction exist in the current TECRDB_compounds_data_dict,
        if not we need to update, calculate pKas and different protonation states of the compound
        """
        #
        self.to_calc_cids = []
        for cur_rxn_dict in self.rxn2calc_dicts:
            self.to_calc_cids += cur_rxn_dict.keys()
        self.to_calc_cids = list(set(self.to_calc_cids))
        self.new_compound_id_list = []; self.new_compound_smiles_list = []
        for cid in self.to_calc_cids:
            if cid not in self.TECRDB_compound_ids:
                self.new_compound_id_list.append(cid)
                try:
                    self.new_compound_smiles_list.append(self.cpd_molstring_dict[cid])
                except KeyError:
                    #it is necessary to provide the molstring for new compounds in order for the calculation to proceed
                    sys.exit("Missing molstring for %s" %cid)
        if self.new_compound_id_list != []:
            new_cpds_info_dict, new_cpds_pH7_sid_dict = group_contribution.get_compound_species_dict(self.new_compound_id_list, self.new_compound_smiles_list)
            self.TECRDB_compounds_data_dict.update(new_cpds_info_dict)
            self.TECRDB_compounds_pH7_species_id_dict.update(new_cpds_pH7_sid_dict)
            self.TECRDB_compound_ids += self.new_compound_id_list
    
    def _update_cpd_groups(self):
        """
        Update the groups in TECRDB_compounds_data_dict if necessary and add groups for new compounds pH 7 species into TECRDB_compounds_data_dict
        Originally 179 groups + self.nondecompose_sids + self.inorg_cpd_with_thermo_data + self.cofactor_cids
        must run _update_cpd_data_dict first
        """
        if self.new_compound_id_list != []:
            self.new_cpd_pH7_sid_list = [self.TECRDB_compounds_pH7_species_id_dict[cid] for cid in self.new_compound_id_list]
            new_cpd_pH7_sid_smiles_list = [self.TECRDB_compounds_data_dict[sid]['smiles_form'] for sid in self.new_cpd_pH7_sid_list]
            new_cpd_pH7_sid_group_mat = compound_groups.get_group_matrix(new_cpd_pH7_sid_smiles_list)
            new_cpd_group_num = new_cpd_pH7_sid_group_mat.shape[1]
            
            for i, sid in enumerate(self.new_cpd_pH7_sid_list):
                cur_sid_group = list(new_cpd_pH7_sid_group_mat[i])
                #add 0 entries for species that are treated as a single group in TECRDB
                updated_sid_group = cur_sid_group[:self.group_only_num] + [0.0] * (self.group_def_num - self.group_only_num) + cur_sid_group[self.group_only_num:]
                self.TECRDB_compounds_data_dict[sid]['updated_groups'] = updated_sid_group

            #if there are new compounds that are nondecomposable, we have to fix the groups in TECRDB_compounds_data_dict
            #otherwise if num_of_groups_for_nondecomposable_cpds = 0, the updated_groups will be the same
            nondecompose_cpd_num = new_cpd_group_num - self.group_only_num
            self.group_def_num = self.group_def_num + nondecompose_cpd_num #now include newly added nondecomposable compound as the group

            for sid in self.TECRDB_compounds_data_dict.keys():
                #update the groups for compounds originally in TECRDB_compounds_data_dict
                if sid not in self.new_cpd_pH7_sid_list:
                    cur_sid_info = self.TECRDB_compounds_data_dict[sid]
                    if 'updated_groups' in cur_sid_info.keys():
                        self.TECRDB_compounds_data_dict[sid]['updated_groups'] = self.TECRDB_compounds_data_dict[sid]['updated_groups'] + [0.0] * nondecompose_cpd_num
                    elif 'updated_groups' not in cur_sid_info.keys() and 'groups' in cur_sid_info.keys():
                        self.TECRDB_compounds_data_dict[sid]['updated_groups'] = self.TECRDB_compounds_data_dict[sid]['groups'] + [0.0] * nondecompose_cpd_num

        else:
            self.new_cpd_pH7_sid_list = []
    
    def _update_cpd_data_and_groups(self):
        self._update_cpd_data_dict()
        self._update_cpd_groups()
        
    def get_species_group_matrix(self):
        """
        Set up the group matrix for training species
        :return: a numpy array with row corresponding to species id and column corresponding to groups
        """
        training_sid_group_mat = []
        for cur_sid in self.training_species_ids:
            cur_sid_info_dict = self.TECRDB_compounds_data_dict[cur_sid]
            if 'updated_groups' in cur_sid_info_dict.keys():
                cur_sid_groups = list(cur_sid_info_dict['updated_groups'])
            else:
                cur_sid_groups = list(cur_sid_info_dict['groups'])
            training_sid_group_mat.append(cur_sid_groups)
        training_sid_group_mat = np.array(training_sid_group_mat)
        return training_sid_group_mat
        
    
    def calc_dGr(self, rxn2calc, pH, IS, T, cpd_molstring_dict = {}, metal_conc_dict = {}, T_correction = True):
        """
        Calculate dGr of reaction using group contribution
        :param rxn2calc: a list of reaction formula, such as ['2 CHB_16761 = CHB_15422 + CHB_16027']
                         a list of dictionary is also possible: [{'CHB_15422':1, 'CHB_16027': 1, 'CHB_16761':-2}]
        :param pH: pH, can be a value or a list of values that matches the number of reactions to calculate
        :param IS: ionic strength, can be a value or a list of values that matches the number of reactions to calculate
        :param T: temperature, can be a value or a list of values that matches the number of reactions to calculate
        :param cpd_molstring_dict: a dictionary with key being compound id and value being molstring, necessary for new compounds in TECRDB
        :param metal_conc_dict: concentration of metal ions, with key being metal ion, value being the concentration
               can be a dict or a list of dict that matches the number of reactions to calculate
               possible metal ions include: 'Mg', 'Ca', 'Co', 'IS', 'K', 'Li', 'Mn', 'Na', 'T', 'Zn'
               note this only works for compounds in TECRDB with metal bound form
        :return: dGr_prime under given condition
        """
        if type(rxn2calc[0]) is dict:
            self.rxn2calc_dicts = rxn2calc
        else:
            self.rxn2calc_dicts = self.formula2dict(rxn2calc)
        self.cpd_molstring_dict = cpd_molstring_dict
        #If there are new compounds, append columns to S matrix transpose
        self._update_cpd_data_and_groups()
        addition_training_species_ids = list(set([self.TECRDB_compounds_pH7_species_id_dict[cid] for cid in self.to_calc_cids])-set(self.training_species_ids))
        self.training_species_ids = self.training_species_ids + addition_training_species_ids
        self.training_S_mat_T = np.hstack((self.training_S_mat_T, np.zeros((self.training_S_mat_T.shape[0], len(addition_training_species_ids)))))
        thermo_transform = thermodynamic_transformations(self.TECRDB_compounds_data_dict, self.TECRDB_compounds_pH7_species_id_dict)
        
        #set up to_calc_S_mat_transpose, row is number of rxns to calculate, column is all_sids
        self.to_calc_S_mat_T = []; self.to_calc_ddGr = np.zeros(len(self.rxn2calc_dicts))
        for i, cur_rxn_dict in enumerate(self.rxn2calc_dicts):
            cur_rxn_stoich_list = [0.0] * len(self.training_species_ids)
            for cid, stoich in cur_rxn_dict.iteritems():
                cur_pH7_sid = self.TECRDB_compounds_pH7_species_id_dict[cid]
                if type(pH) == list:
                    try:
                        assert len(pH) == len(IS) == len(T) == len(rxn2calc)
                    except AssertionError:
                        print 'Number of conditions should match with number of reactions!'
                    if metal_conc_dict == {}:
                        cur_cpd_ddGf = thermo_transform._ddGf_pH7_num(cid, pH[i], IS[i], T[i], {})
                    else:
                        cur_cpd_ddGf = thermo_transform._ddGf_pH7_num(cid, pH[i], IS[i], T[i], metal_conc_dict[i])
                else:
                    cur_cpd_ddGf = thermo_transform._ddGf_pH7_num(cid, pH, IS, T, metal_conc_dict)
                self.to_calc_ddGr[i] += cur_cpd_ddGf * stoich
                cur_species_pos = self.training_species_ids.index(cur_pH7_sid)
                cur_rxn_stoich_list[cur_species_pos] = stoich
            self.to_calc_S_mat_T.append(cur_rxn_stoich_list)
        self.to_calc_S_mat_T = np.array(self.to_calc_S_mat_T)
        
        self.training_sid_group_mat = self.get_species_group_matrix()
        
        #set up training matrices and obtain estimates for group contribution (component contribution)
        self.training_rxn_group_mat = np.dot(self.training_S_mat_T, self.training_sid_group_mat)
        self.training_inv_S, self.training_P_R_rc, self.training_P_N_rc = group_contribution.invert_matrix(self.training_S_mat_T.T)
        self.training_inv_GS, self.training_P_R_gc, self.training_P_N_gc = group_contribution.invert_matrix(self.training_rxn_group_mat)
        
        self.training_dG0_rc = np.dot(self.training_inv_S.T, self.training_dGr0s)
        self.training_dG0_gc = np.dot(np.dot(self.training_sid_group_mat, self.training_inv_GS),self.training_dGr0s)
        self.training_dG0_cc = np.dot(self.training_P_R_rc, self.training_dG0_rc) + np.dot(self.training_P_N_rc, self.training_dG0_gc)
        
        #calculate dGr0 and dGr_prime values
        self.predicted_dGr0 = np.dot(self.to_calc_S_mat_T, self.training_dG0_cc)
        self.predicted_dGr_prime = self.predicted_dGr0 + self.to_calc_ddGr
        if T == 298.15 or (len(np.unique(T)) == 1 and np.unique(T)[0] == 298.15) or T_correction == False:
            return self.predicted_dGr_prime
        else:
            self.predicted_dSr0 = np.array(self.dSr_calc.dSr_predictor(self.rxn2calc_dicts, \
                                           self.TECRDB_compounds_data_dict, self.TECRDB_compounds_pH7_species_id_dict))
            if type(T) == list:
                self.predicted_dGr_prime_T_corr = self.predicted_dGr_prime - np.multiply(np.array(T) - 298.15, self.predicted_dSr0)
            else:
                self.predicted_dGr_prime_T_corr = self.predicted_dGr_prime - (T - 298.15)*self.predicted_dSr0
            return self.predicted_dGr_prime_T_corr
        
        
    @staticmethod
    def get_compound_species_dict(compound_id_list, molstring_list):
        """
        Calculate pKas of the compound, as well as H number and charge of its protonation states
        :param compound_id_list: the list of compound id
        :param molstring_list: the list of molstring for the compound
        :return: compound_species_info_dict: key is species id, values is dictionary of the species info
                 compound_pH7_species_id_dict: key is compound_id and value is id of its pH7 species
        """
        compound_species_info_dict = {}
        compound_pH7_sid_dict = {}
        for i, cur_compound_id in enumerate(compound_id_list):
            print "Calculating pKas and protonation states for %s" % cur_compound_id
            cur_smiles_form = molstring_list[i]
            pKas, nHs, zs, pH7_species_pos, pH7_species_smiles = Get_pKas_Hs_zs_pH7smiles(cur_smiles_form)
            binding_constants = np.cumsum([0] + pKas)
            for i, cur_z in enumerate(zs):
                cur_sid = cur_compound_id + '_' + str(int(cur_z))
                compound_species_info_dict[cur_sid] = {'H_number': nHs[i], 'charge':cur_z, 'binding_constant': binding_constants[i], 'compound_id': cur_compound_id}
                if i == pH7_species_pos:
                    compound_species_info_dict[cur_sid]['smiles_form'] = pH7_species_smiles
                    compound_pH7_sid_dict[cur_compound_id] = cur_sid
        return compound_species_info_dict, compound_pH7_sid_dict
    
    @staticmethod
    def invert_matrix(A, eps=1e-10, method = 'numpy'):
        """
        Calculate the inverse of the matrix, possible with different modules
        """
        n, m = A.shape
        if method == 'numpy':
            U, s, V_H = np.linalg.svd(A, full_matrices=True)
            V = V_H.T
            r = sum(abs(s) > eps)
            inv_S = np.matrix(np.diag([1.0/s[i] for i in xrange(r)]))
            inv_A = V[:, :r] * inv_S * U[:, :r].T
            P_R   = np.dot(U[:, :r], U[:, :r].T)
            P_N   = np.eye(n) - P_R
        elif method == 'nosvd':
            inv_A = np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + np.eye(n)*4e-6).T)
            # then the solution for (A.T * x = b) will be given by (x = inv_A.T * b)
            P_R = np.dot(A, inv_A)
            P_N = np.eye(n) - P_R
            r = sum(np.abs(np.linalg.eig(P_R)[0]) > 0.5)
        return np.array(inv_A), np.array(P_R), np.array(P_N)