import numpy as np
from process_thermo_data import thermo_data
from process_TECRDB_compounds import TECRDB_compounds_data
from dSr_calculation import dSr_calculation
from thermodynamic_transformations import thermodynamic_transformations

class training_data(object):
    def __init__(self, metal_correction=False, T_correction=True):
        """
        A module that prepares the training data for dGr and dGf calculations
        :param metal_correction: whether to perform correction on metal concentration
        :param T_correction: whether to perform correction on temperature
        """
        #load the compound data and thermo data
        self.get_thermo_data()
        self.get_TECRDB_compound_data()
        self.metal_correction = metal_correction #metal correction shown not to help with the estimate globally, so currently set to False
        self.T_correction = T_correction
        
    def get_thermo_data(self):
        """
        Get compound thermodynamic data and reaction thermodynamic data
        """
        thermodynamics_data = thermo_data()
        thermodynamics_data.get_thermo_data()
        self.all_thermo_data_dict = thermodynamics_data.all_thermo_data_dict
    
    def get_TECRDB_compound_data(self):
        """
        Get information of compounds in TECRDB
        """
        TECRDB_cpd_data = TECRDB_compounds_data()
        self.TECRDB_compounds_data_dict = TECRDB_cpd_data.TECRDB_compounds_data_dict
        self.TECRDB_compounds_pH7_species_id_dict = TECRDB_cpd_data.TECRDB_compounds_pH7_species_id_dict
        self.TECRDB_compounds_least_H_sid_dict = TECRDB_cpd_data.TECRDB_compounds_least_H_sid_dict
        #we will fit the cofactors as a single group rather than get them decomposed
        self.cofactor_cids = ['CHB_16048', 'CHB_17621', 'PBC_4481', 'PBC_4066204', 'CHB_16238', 'CHB_17877', 'CHB_15846', 'CHB_16908', 'CHB_18009', 'CHB_16474', 'PBC_123926', 'MAN_10059', 'PBC_4156341', 'MAN_10151']
        self.inorg_cpd_with_thermo_data = [] #inorganic compounds that already have dG_f, dH_f etc available, we will treat them as a single group
        for sid in self.TECRDB_compounds_data_dict.keys():
            if 'dG_f' in self.TECRDB_compounds_data_dict[sid].keys():
                cur_cid = self.TECRDB_compounds_data_dict[sid]['compound_id']
                if cur_cid not in self.inorg_cpd_with_thermo_data:
                    self.inorg_cpd_with_thermo_data.append(cur_cid)
        self.inorg_cpd_with_thermo_data.sort()
    
    def get_TECRDB_dSr_data(self):
        """
        Get dSr for reactions in TECRDB, return a dictionary with key being reaction id, value being its dSr
        """
        dSr_calc = dSr_calculation(self.TECRDB_compounds_data_dict, self.TECRDB_compounds_pH7_species_id_dict)
        self.TECRDB_rxn_dSr_dict = dSr_calc.get_TECRDB_rxn_dSr_dict(self.all_thermo_data_dict['dG_r'])
            
    def get_training_data(self):
        """
        Get training dGr and dGf data ids, as well as the dominant pH 7 species ids for compounds involved in those data points
        """
        #For reaction with Keq data
        self.TECRDB_rxn_ids = sorted(self.all_thermo_data_dict['dG_r'].keys())
        TECRDB_sids = []
        for rid in self.TECRDB_rxn_ids:
            rid_data_dict = self.all_thermo_data_dict['dG_r'][rid]
            cpds_in_rxn = rid_data_dict['rxn_dict'].keys()
            for cpd in cpds_in_rxn:
                cpd_pH7_sid = self.TECRDB_compounds_pH7_species_id_dict[cpd]
                TECRDB_sids.append(cpd_pH7_sid)
        
        #For inorganic compounds with dG_f data
        dG_f_inorg_cpd_sids = []
        dG_f_inorg_cpd_data_name = []
        for cid in self.inorg_cpd_with_thermo_data:
            cpd_pH7_sid = self.TECRDB_compounds_pH7_species_id_dict[cid]
            dG_f_inorg_cpd_sids.append(cpd_pH7_sid)
            dG_f_inorg_cpd_data_name.append('dG_f_cpd#' + cpd_pH7_sid)
            
        #For other compounds with dG_f data
        dG_f_cpd_thermo_data_sids = []
        dG_f_cpd_data_name = []
        '''
        for species_id in sorted(self.all_thermo_data_dict['dG_f'].keys()):
            if species_id in self.TECRDB_compounds_data_dict.keys() and self.TECRDB_compounds_data_dict[species_id]['compound_id'] not in (self.inorg_cpd_with_thermo_data + self.cofactor_cids):
                dG_f_cpd_data_name.append('dG_f_cpd#' + species_id)
                dG_f_cpd_thermo_data_sids.append(species_id)
        for species_id in sorted(self.all_thermo_data_dict['dG_f_prime'].keys()):
            if species_id in self.TECRDB_compounds_data_dict.keys() and self.TECRDB_compounds_data_dict[species_id]['compound_id'] not in (self.inorg_cpd_with_thermo_data + self.cofactor_cids):
                dG_f_cpd_data_name.append('dG_f_prime_cpd#' + species_id)
                dG_f_cpd_thermo_data_sids.append(species_id)
        '''
        
        self.training_species_ids = sorted(list(set(TECRDB_sids + dG_f_inorg_cpd_sids + dG_f_cpd_thermo_data_sids)))
        self.training_rxn_ids = self.TECRDB_rxn_ids + dG_f_inorg_cpd_data_name + dG_f_cpd_data_name
        #print len(TECRDB_rxn_ids)
        #print len(dG_f_inorg_cpd_data_name)
        
    def get_training_S_matrix(self):
        """
        #following get_training_data function
        Get the stoichiometric matrix for training data, with row corresponding to species ids and column corresponding to reaction ids.
        Also calculate dGr0 and dGf0 for the data, using functions in module thermodynamic_transformations
        :return: - self.training_S_mat_T: transpose of the stoichiometric matrix
                 - self.training_dGr0s: a vector of dGr0/dGf0 for reactions/compounds (compound is treated as equivalent to the reaction with 1 in its corresponding column of S)
        """
        thermo_transform = thermodynamic_transformations(self.TECRDB_compounds_data_dict, self.TECRDB_compounds_pH7_species_id_dict)
        self.get_TECRDB_dSr_data()
        training_S_mat_T = []
        training_dGr0s = []

        #for data on reactions
        for cur_rid in self.training_rxn_ids[:len(self.TECRDB_rxn_ids)]:
            cur_rxn_stoich_list = [0.0] * len(self.training_species_ids)
            cur_rxn_dict = self.all_thermo_data_dict['dG_r'][cur_rid]['rxn_dict']
            cur_dGr0 = thermo_transform._get_dGr0_num(self.all_thermo_data_dict['dG_r'], \
                       self.TECRDB_rxn_dSr_dict, cur_rid, metal_correction=self.metal_correction, T_correction=self.T_correction)
            training_dGr0s.append(cur_dGr0)

            for compound_id, stoich in cur_rxn_dict.iteritems():
                cur_pH7_sid = self.TECRDB_compounds_pH7_species_id_dict[compound_id]
                cur_sid_pos = self.training_species_ids.index(cur_pH7_sid)
                cur_rxn_stoich_list[cur_sid_pos] = stoich
            training_S_mat_T.append(cur_rxn_stoich_list)

        #for data on compounds
        for cur_rid in self.training_rxn_ids[len(self.TECRDB_rxn_ids):]:
            cur_rxn_stoich_list = [0.0] * len(self.training_species_ids)
            cur_sid = cur_rid.split('#')[1]
            cur_sid_pos = self.training_species_ids.index(cur_sid)
            cur_rxn_stoich_list[cur_sid_pos] = 1.0 #for a single sid
            training_S_mat_T.append(cur_rxn_stoich_list)

            if 'dG_f_cpd' in cur_rid:
                try:
                    #for inorganic species
                    dGf0 = self.TECRDB_compounds_data_dict[cur_sid]['dG_f']
                except KeyError:
                    #for organic species
                    dGf0 = self.all_thermo_data_dict['dG_f'][cur_sid]
                training_dGr0s.append(dGf0)
            if 'dG_f_prime_cpd' in cur_rid:
                dGf_prime_val = self.all_thermo_data_dict['dG_f_prime'][cur_sid]
                dGf0 = thermo_transform._get_dGf0_num(cur_sid, dGf_prime_val)
                training_dGr0s.append(dGf0)
        
        self.training_S_mat_T = np.array(training_S_mat_T)
        self.training_dGr0s = np.array(training_dGr0s)
        print "Calculated %d dGr0s with %d compounds as training data" %(self.training_S_mat_T.shape[0], self.training_S_mat_T.shape[1])
        
    def get_training_data_and_matrix(self):
        self.get_training_data()
        self.get_training_S_matrix()