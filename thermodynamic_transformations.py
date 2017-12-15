import numpy as np
from sympy import symbols
from metal_temperature_transform import metal_T_transform_dG_f
from scipy.special import logsumexp
R = 8.3144621 #J/K/mol

class thermodynamic_transformations(object):
    """
    A module that calculates thermodynamic transformation of dGf and dGr as a function of pH, T, ionic strength and metal ion concentrations.
    """
    def __init__(self, compounds_data_dict, pH7species_id_dict):
        self.compounds_data_dict = compounds_data_dict
        self.pH7species_id_dict = pH7species_id_dict
        self.get_least_protonated_sids()
        
    def get_least_protonated_sids(self):
        """
        Get the least protonated state species id for each compound id
        :return: a dictionary with key being compound_id, value being species id for the least protonated state
        """
        self.cid_to_least_H_sid_dict = {}
        for sid in self.compounds_data_dict.keys():
            cur_sid_info_dict = self.compounds_data_dict[sid]
            if cur_sid_info_dict['binding_constant'] == 0.0 and 'metal_type' not in cur_sid_info_dict.keys():
                self.cid_to_least_H_sid_dict[cur_sid_info_dict['compound_id']] = sid

    @staticmethod
    def debye_huckel_dG_f(IS, T):
        """
        debye huckel correction on dG_f as a function of ionic strength and temperature
        :param IS: ionic strength
        :param T: temperature
        :return: float to be used in for correction on dG_f
        """
        alpha_dG = (1e-3*(9.20483*T) - 1e-5*(1.284668 * T**2) + 1e-8*(4.95199 * T**3))*1000
        beta = 1.6
        IS_sqrt = np.power(IS,0.5)
        if IS < np.power(10,-2.3):
            debye_huckel = alpha_dG * IS_sqrt
        elif IS <= 0.1:
            debye_huckel = alpha_dG * IS_sqrt / (1 + beta * IS_sqrt)
        else:
            debye_huckel = alpha_dG * (IS_sqrt/(1 + IS_sqrt) - 0.3 * IS) #davies equation at ionic strenght greater than 0.1
        return debye_huckel
    
    @staticmethod
    def debye_huckel_dH_f(IS, T):
        """
        debye huckel correction on dH_f as a function of ionic strength and temperature
        :param IS: ionic strength
        :param T: temperature
        :return: float to be used in for correction on dH_f
        """
        alpha_dH = (-1e-5*(1.28466*T**2) + 1e-8*(9.90399 * T**3))*1000
        beta = 1.6
        IS_sqrt = np.power(IS,0.5)
        if IS < np.power(10,-2.3):
            debye_huckel = alpha_dH * IS_sqrt
        elif IS <= 0.1:
            debye_huckel = alpha_dH * IS_sqrt / (1 + beta * IS_sqrt)
        else:
            debye_huckel = alpha_dH * (IS_sqrt/(1 + IS_sqrt) - 0.3 * IS) #davies equation at ionic strenght greater than 0.1
        return debye_huckel

    def _ddGf_least_H_state_num(self, compound_id, pH, IS, T, metal_conc_dict):
        """
        Calculate the difference in dG_f between reactant and its least protonated state
        :param compound_id: compound_id in TECRDB
        :param pH: pH
        :param IS: ionic strength
        :param T: temperature
        :param metal_conc_dict: a dictionary with key being metal ion, value being the respective concentration
        :return: difference in dG_f between reactant and its least protonated state
        """
        sid_list = [sid for sid in self.compounds_data_dict.keys() if self.compounds_data_dict[sid]['compound_id'] == compound_id]
        least_H_sid = self.cid_to_least_H_sid_dict[compound_id]
        ddGf_list = []
        used_sid_list = []
        for sid in sid_list:
            if 'metal_type' in self.compounds_data_dict[sid].keys():
                #write ddGf for metal bound species, which is dG'(metal bound species) - dG_0(least H species)
                try:
                    metal_conc = metal_conc_dict[self.compounds_data_dict[sid]['metal_type']]; pMetal = -np.log10(metal_conc)
                    metal_number = self.compounds_data_dict[sid]['metal_number']; metal_binding_constant = self.compounds_data_dict[sid]['binding_constant']

                    cur_ddGf = - R * 298.15 * np.log(10) * metal_binding_constant\
                    + pH * self.compounds_data_dict[sid]['H_number'] * R * T * np.log(10) + pMetal * metal_number * R * T * np.log(10)\
                    - thermodynamic_transformations.debye_huckel_dG_f(IS, T) * (self.compounds_data_dict[sid]['charge']**2 - self.compounds_data_dict[sid]['H_number'])\
                    - metal_number * metal_T_transform_dG_f(self.compounds_data_dict[sid]['metal_type'], T)

                    ddGf_list.append(cur_ddGf); used_sid_list.append(sid)
                except KeyError: #the metal is not present in media condition
                    pass
            else:
                try:
                    #for inorganic compounds we directly use dG_f data
                    ddGf0 = self.compounds_data_dict[sid]['dG_f'] - self.compounds_data_dict[least_H_sid]['dG_f']
                except KeyError:
                    # next write equations for species at different protonation states, the energy_diff is dG'(species) - dG_0(least H species)
                    proton_binding_constant = self.compounds_data_dict[sid]['binding_constant']
                    ddGf0 = - R * 298.15 * np.log(10) * proton_binding_constant

                cur_ddGf = ddGf0 + pH * self.compounds_data_dict[sid]['H_number'] * R * T * np.log(10)\
                - thermodynamic_transformations.debye_huckel_dG_f(IS, T) * (self.compounds_data_dict[sid]['charge']**2 - self.compounds_data_dict[sid]['H_number'])

                ddGf_list.append(cur_ddGf); used_sid_list.append(sid)

        #Now calculate dG'(compound) - dG_0(least H species)
        if len(ddGf_list) == 1:
            ddGf_prime = ddGf_list[0] #energy difference of the species is equivalent to that of the compound since there is only one species
            species_fraction = [1.0]    
        else:
            ddGf_prime = -R*T*logsumexp(np.array(ddGf_list)/(-R*T))
            species_fraction = [np.exp((ddGf_prime - energy_diff)/R/T) for energy_diff in ddGf_list]

        return ddGf_prime, species_fraction, used_sid_list
    
    def _ddGf_pH7_num(self, compound_id, pH, IS, T, metal_conc_dict):
        """
        Calculate the difference in dG_f between reactant and the dominant protonation state at pH 7
        :param compound_id: compound_id in TECRDB
        :param pH: pH
        :param IS: ionic strength
        :param T: temperature
        :param metal_conc_dict: a dictionary with key being metal ion, value being the respective concentration
        :return: difference in dG_f between reactant and its dominant protonation state at pH 7
        """
        pH7_sid = self.pH7species_id_dict[compound_id]
        #energy difference dG_0(pH7 species) - dG_0(least protonated species)
        pH7_sid_binding_constant = self.compounds_data_dict[pH7_sid]['binding_constant']
        ddGf_pH7_and_least_H = - R * 298.15 * np.log(10) * pH7_sid_binding_constant
        #energy difference dG'(compound) - dG_0(least protonated state)
        ddGf_prime_least_H, _, _ = self._ddGf_least_H_state_num(compound_id, pH, IS, T, metal_conc_dict)
        #energy difference dG'(compound) - dG_0(pH7 species)
        ddGf_prime_pH7 = ddGf_prime_least_H - ddGf_pH7_and_least_H
        return ddGf_prime_pH7
    
    def _get_dGr0_num(self, Keq_data_dict, TECRDB_rxn_dSr_dict, rid, metal_correction=False, T_correction=True):
        """
        Transform dGr prime to dGr standard for protonation states at pH 7
        dGr standard = dGr prime - ddGr, where ddGr is the sum of ddGf calculated by _ddGf_pH7_num
        :param Keq_data_dict: generated from module process_thermo_data, the dictionary that stores thermodynamic measurements of reactions
                              key being reaction id, value being the info of reaction measurement
        :param TECRDB_rxn_dSr_dict: generated from module dSr_calculation, the dictionary with key being reaction id, and value being dSr of the reaction
        :param rid: reaction id whose dGr_prime to be transformed to dGr standard
        :param metal_correction: whether to perform correction on metal concentration
        :param T_correction: whether to perform correction on temperature
        :return: dGr standard for protonation states at pH 7
        """
        reaction_info_dict = Keq_data_dict[rid]
        pH = reaction_info_dict['pH'];IS = reaction_info_dict['IS'];T = reaction_info_dict['T'];Keq = reaction_info_dict['Keq']
        rxn_dict = reaction_info_dict['rxn_dict']
        if metal_correction == True:
            metal_conc_dict = reaction_info_dict['metal ions']
        else:
            metal_conc_dict = {}
        #we can only correct for Mg concentration
        #if 'Mg' in metal_conc_dict.keys():
        #    metal_conc_dict = {'Mg': metal_conc_dict['Mg']}
        #else:
        #    metal_conc_dict = {}
        dGr_prime = -R*T*np.log(Keq)
        ddGr = 0
        for compound_id, stoich in rxn_dict.iteritems():        
            cur_ddGf = self._ddGf_pH7_num(compound_id, pH, IS, T, metal_conc_dict)
            ddGr += cur_ddGf * stoich
            
        if T_correction == True:
            dGr0 = dGr_prime - ddGr + (T - 298.15) * TECRDB_rxn_dSr_dict[rid]
        else:
            dGr0 = dGr_prime - ddGr

        return dGr0
    
    def _get_dGf0_num(self, species_id, dG_f_prime):
        pH = 7.0; IS = 0.25; T = 298.15 #the data are from defined conditions
        ddGf = pH * self.compounds_data_dict[species_id]['H_number'] * R * T * np.log(10)\
        - thermodynamic_transformations.debye_huckel_dG_f(IS, T) * (self.compounds_data_dict[species_id]['charge']**2 - self.compounds_data_dict[species_id]['H_number'])
        dG_f_standard = dG_f_prime - ddGf
        return dG_f_standard
    
    ####################################################################################################################
    """
    All functions above are sufficient for necessary thermodynamic transformations, the functions below are for setting 
    up symbolic equations to optimize binding constants. Note the compounds here must have already been present in 
    self.compounds_data_dict
    """
    
    def setup_symbols_for_species_pKs(self, sid_list):
        """
        Set up the dictionary to store symbols and its original values for the pK variables
        Additionally, write the sequence of ion bound states from a given species id to the least protonated species id,
        such sequence will be helpful to write binding polynomial.
        """
        new_variable_index = 0
        self.variable_vector_dict = {}
        for species_id in sid_list:
            pK_data_val = self.get_pK_val(species_id)  
            self.variable_vector_dict[species_id] = [symbols('x[%d]'%new_variable_index), pK_data_val]
            new_variable_index += 1
        #for each species_id, set up the sequence of species that eventually lead to least protonated state, for binding constant calculation
        self.compounds_species_id_sequence = {}
        for species_id in self.compounds_data_dict.keys():
            self.compounds_species_id_sequence[species_id] = self.get_sequence_of_species_ids(species_id)
            
    def get_unbound_species_id(self, species_id):
        """
        Get the species id for the unbound state, for non metal species, we get -1 H protonated states,
        for metal species, we get -1 metal bound states
        """
        if species_id in self.cid_to_least_H_sid_dict.values(): #already the least protonated state
            return
        else:
            compound_id = self.compounds_data_dict[species_id]['compound_id']
            if 'metal_type' not in self.compounds_data_dict[species_id].keys():#non metal compounds
                if '-5b' in species_id:#in the case of tautomers
                    unbound_species_id = compound_id + '_-5a'
                    return unbound_species_id
                else:
                    cur_species_charge = int(self.compounds_data_dict[species_id]['charge'])
                    minus1_species_charge = cur_species_charge - 1
                    minus1_species_id = compound_id + '_' + str(minus1_species_charge)
                    if species_id[-1] == 'a' or species_id[-1] == 'b': #in the case of tautomer
                        minus1_species_id += species_id[-1]
                    return minus1_species_id
            else:#metal compounds
                if self.compounds_data_dict[species_id]['metal_number'] == 1:#one metal bound
                    non_metal_species_id = compound_id + '_' + species_id.split('_')[2] #compound_id + charge get the non metal species
                    return non_metal_species_id
                else:
                    cur_metal_type = self.compounds_data_dict[species_id]['metal_type']; cur_metal_num = int(self.compounds_data_dict[species_id]['metal_number'])
                    species_id_minus1_metal = compound_id + '_' + species_id.split('_')[2] + '_' + cur_metal_type + str(cur_metal_num-1) + '_L1'
                    return species_id_minus1_metal
    
    def get_pK_val(self, species_id):
        """
        Get disassociation constant for the given species id. pK value can be either pKa or pKmetal,
        depending on whether it is proton bound or metal bound
        """
        if species_id in self.cid_to_least_H_sid_dict.values():
            return
        else:
            unbound_species_id = self.get_unbound_species_id(species_id)
            pK_val = self.compounds_data_dict[species_id]['binding_constant'] - self.compounds_data_dict[unbound_species_id]['binding_constant']
            return pK_val
     
    def get_sequence_of_species_ids(self, species_id):
        """
        Get the sequence of species_ids that eventually lead to least protonated species id,
        such sequence is used later to calculate binding polynomial
        """
        sequence_of_species_ids = [species_id]
        if species_id in self.cid_to_least_H_sid_dict.values():
            return sequence_of_species_ids
        else:
            unbound_species_id = self.get_unbound_species_id(species_id)
            sequence_of_species_ids += self.get_sequence_of_species_ids(unbound_species_id)
            return sequence_of_species_ids
        
    def get_binding_constant(self, species_id, write_pK_as_variable = False):
        """
        Get the binding polynomial of the ion bound state with respect to its least protonated form
        :param write_pK_as_variable: if true return float, else return sympy symbol object
        """
        cur_sequence_of_species_ids = self.compounds_species_id_sequence[species_id]
        cur_binding_constant = 0.0
        for sid in cur_sequence_of_species_ids:
            if sid in self.cid_to_least_H_sid_dict.values():#least protonated state has binding constant 0
                pass
            elif sid in self.variable_vector_dict.keys() and write_pK_as_variable == True: #write symbolic pK value to add in binding constants
                cur_binding_constant += self.variable_vector_dict[sid][0]
            else: #just get the numerical pK value
                cur_binding_constant += self.get_pK_val(sid)
        return cur_binding_constant
    
    def _ddGf_least_H_state_sym(self, compound_id, pH, IS, T, metal_conc_dict):
        """
        Calculate the difference in dG_f between reactant and its least protonated state, return symbolic expression with pK value as variable
        :param compound_id: compound_id in TECRDB
        :param pH: pH
        :param IS: ionic strength
        :param T: temperature
        :param metal_conc_dict: a dictionary with key being metal ion, value being the respective concentration
        :return: symbolic expression for difference in dG_f between reactant and its least protonated state, the variable is pK in self.variable_vector_dict
        """
        sid_list = [sid for sid in self.compounds_data_dict.keys() if self.compounds_data_dict[sid]['compound_id'] == compound_id]
        least_H_sid = self.cid_to_least_H_sid_dict[compound_id]
        ddGf_list = []
        used_sid_list = []
        for sid in sid_list:
            if 'metal_type' in self.compounds_data_dict[sid].keys():
                #write ddGf for metal bound species, which is dG'(metal bound species) - dG_0(least H species)
                try:
                    metal_conc = metal_conc_dict[self.compounds_data_dict[sid]['metal_type']]; pMetal = -np.log10(metal_conc)
                    metal_number = self.compounds_data_dict[sid]['metal_number']
                    #the binding polynomial is expressed in symbolic form with pK written as variable
                    metal_binding_constant = self.get_binding_constant(sid, write_pK_as_variable = True)

                    cur_ddGf = - R * 298.15 * np.log(10) * metal_binding_constant\
                    + pH * self.compounds_data_dict[sid]['H_number'] * R * T * np.log(10) + pMetal * metal_number * R * T * np.log(10)\
                    - thermodynamic_transformations.debye_huckel_dG_f(IS, T) * (self.compounds_data_dict[sid]['charge']**2 - self.compounds_data_dict[sid]['H_number'])\
                    - metal_number * metal_T_transform_dG_f(self.compounds_data_dict[sid]['metal_type'], T)

                    ddGf_list.append(cur_ddGf); used_sid_list.append(sid)
                except KeyError: #the metal is not present in media condition
                    pass
            else:
                try:
                    # for inorganic compounds we directly use dG_f data
                    ddGf0 = self.compounds_data_dict[sid]['dG_f'] - self.compounds_data_dict[least_H_sid]['dG_f'] #for inorganic compounds, proton delta G is 0
                except KeyError:
                    # next write equations for species at different protonation states, the energy_diff is dG'(species) - dG_0(least H species)
                    proton_binding_constant = self.get_binding_constant(sid, write_pK_as_variable = True)
                    ddGf0 = - R * 298.15 * np.log(10) * proton_binding_constant

                cur_ddGf = ddGf0 + pH * self.compounds_data_dict[sid]['H_number'] * R * T * np.log(10)\
                - thermodynamic_transformations.debye_huckel_dG_f(IS, T) * (self.compounds_data_dict[sid]['charge']**2 - self.compounds_data_dict[sid]['H_number'])

                ddGf_list.append(cur_ddGf); used_sid_list.append(sid)

        #we are doing this particular manipulation because for later numerical fitting, exponential can blow up due to large values
        #the mathematical transformation looks like: ln(e^A + e^B + e^C) = C + ln(e^(A-C) + e^(B-C) + 1)
        ddGf_least_H = ddGf_list[used_sid_list.index(least_H_sid)]
        delta_ddGf_list = [energy_diff - ddGf_least_H for energy_diff in ddGf_list]

        # Now calculate dG'(compound) - dG_0(least H species)
        if len(ddGf_list) == 1:
            ddGf_prime = str(ddGf_list[0]) #energy difference of the species is equivalent to that of the compound since there is only one species
        else:
            ddGf_prime = str(ddGf_least_H) + '-' + str(R * T) + '*np.log('
            for k, energy_difference in enumerate(delta_ddGf_list):
                exp_component = str(energy_difference/(-R * T))
                if k == 0:
                    ddGf_prime += 'np.exp(%s)' % exp_component
                else:
                    ddGf_prime += ' + np.exp(%s)' % exp_component
            ddGf_prime += ')'
        return ddGf_prime
    
    def _ddGf_pH7_sym(self, compound_id, pH, IS, T, metal_conc_dict):
        """
        Calculate the difference in dG_f between reactant and the dominant protonation state at pH 7, return symbolic expression with pK value as variable
        :param compound_id: compound_id in TECRDB
        :param pH: pH
        :param IS: ionic strength
        :param T: temperature
        :param metal_conc_dict: a dictionary with key being metal ion, value being the respective concentration
        :return: symbolic expression for difference in dG_f between reactant and its dominant protonation state at pH 7, the variable is pK in self.variable_vector_dict
        """
        pH7_sid = self.pH7species_id_dict[compound_id]
        # energy difference dG_0(pH7 species) - dG_0(least protonated species)
        pH7_sid_binding_constant = self.get_binding_constant(pH7_sid, write_pK_as_variable = True)
        ddGf_pH7_and_least_H = - R * 298.15 * np.log(10) * pH7_sid_binding_constant
        #energy difference dG'(compound) - dG_0(least protonated state)
        ddGf_prime_least_H = self._ddGf_least_H_state_sym(compound_id, pH, IS, T, metal_conc_dict)
        #energy difference dG'(compound) - dG_0(pH7 species)
        ddGf_prime_pH7 = ddGf_prime_least_H + '- (' + str(ddGf_pH7_and_least_H) + ')'
        return ddGf_prime_pH7
    
    def _get_dGr0_sym(self, Keq_data_dict, TECRDB_rxn_dSr_dict, rid, metal_correction=False, T_correction=True):
        """
        Transform dGr prime to dGr standard for protonation states at pH 7, return symbolic expression with pK value as variable
        dGr standard = dGr prime - ddGr, where ddGr is the sum of ddGf calculated by _ddGf_pH7_sym
        :param Keq_data_dict: generated from module process_thermo_data, the dictionary that stores thermodynamic measurements of reactions
                              key being reaction id, value being the info of reaction measurement
        :param TECRDB_rxn_dSr_dict: generated from module dSr_calculation, the dictionary with key being reaction id, and value being dSr of the reaction
        :param rid: reaction id whose dGr_prime to be transformed to dGr standard
        :param metal_correction: whether to perform correction on metal concentration
        :param T_correction: whether to perform correction on temperature
        :return: symbolic expression for dGr standard for protonation states at pH 7, the variable is pK in self.variable_vector_dict
        """
        reaction_info_dict = Keq_data_dict[rid]
        pH = reaction_info_dict['pH'];IS = reaction_info_dict['IS'];T = reaction_info_dict['T'];Keq = reaction_info_dict['Keq']
        rxn_dict = reaction_info_dict['rxn_dict'];dGr_prime = -R*T*np.log(Keq)
        if metal_correction == True:
            metal_conc_dict = reaction_info_dict['metal ions']
        else:
            metal_conc_dict = {}

        ddGr = ''
        for compound_id, stoich in rxn_dict.iteritems():
            cur_ddGf = self._ddGf_pH7_sym(compound_id, pH, IS, T, metal_conc_dict)
            ddGr += '(' + str(stoich) + ')' + '*' + '(' + cur_ddGf + ')' + '+'
        ddGr = ddGr[:-1] #remove + sign in the end

        if T_correction == True:
            dGr0 = str(dGr_prime + (T - 298.15) * TECRDB_rxn_dSr_dict[rid]) + '- (' + ddGr + ')'
        else:
            dGr0 = str(dGr_prime) + '- (' + ddGr + ')'
        return dGr0