import pandas as pd
import re

class thermo_data(object):
    def __init__(self):
        """
        A module that processes compound thermodynamic data and reaction thermodynamic data
        """
        self.all_thermo_data_dict = {}
        
    @staticmethod
    def parse_reaction_formula_side(s):
        """ 
            Parses the side formula, e.g. '2 CHB_15377 + CHB_15422'
            Ignores stoichiometry.

            Returns:
                The set of CIDs.
        """
        if s.strip() == "null":
            return {}

        compound_bag = {}
        for member in re.split('\s+\+\s+', s):
            tokens = member.split(None, 1)
            if len(tokens) == 1:
                amount = 1
                key = member
            else:
                try:
                    amount = float(tokens[0])
                except ValueError:
                    pass
                key = tokens[1]

            try:
                compound_bag[key] = compound_bag.get(key, 0) + amount
            except ValueError:
                pass

        return compound_bag

    @staticmethod
    def parse_formula(formula, arrow='='):
        """ 
            Parses a two-sided formula such as: 2 C00001 => C00002 + C00003 

            Return:
                The set of substrates, products and the direction of the reaction
        """
        tokens = formula.split(arrow)
        if len(tokens) < 2:
            raise ValueError('Reaction does not contain the arrow sign (%s): %s'
                             % (arrow, formula))
        if len(tokens) > 2:
            raise ValueError('Reaction contains more than one arrow sign (%s): %s'
                             % (arrow, formula))

        left = tokens[0].strip()
        right = tokens[1].strip()

        sparse_reaction = {}
        for cid, count in thermo_data.parse_reaction_formula_side(left).iteritems():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count 

        for cid, count in thermo_data.parse_reaction_formula_side(right).iteritems():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count 

        return sparse_reaction
    
    def get_compound_data(self, file_name):
        """
        Extract thermodynamic data for compounds, including heat capacity (Cp), formation entropy (dS_f), formation enthalpy (dH_f_, formation energy (dG_f), 
        and transformed formation energy at pH 7, ionic strength 0.25 and 298.15 K

        Write into a dictionary that stores categories of thermodynamic data for different compounds
        """
        self.all_thermo_data_dict['Cp'] = {}
        self.all_thermo_data_dict['dS_f'] = {}
        self.all_thermo_data_dict['dH_f'] = {}
        self.all_thermo_data_dict['dG_f'] = {}
        self.all_thermo_data_dict['dG_f_prime'] = {}
        compounds_thermo_data = pd.read_csv(file_name)
        for i, row in compounds_thermo_data.iterrows():
            self.all_thermo_data_dict[row['data type']][str(row['updated_species_id'])] = float(row['value'])
        
    def get_TECRDB_rxn_data(self, file_name):
        """
        Extract thermodynamic data for reactions, including transformed reaction energy (dG_r), transformed reaction enthalpy (dH_r)

        Write into a dictionary that stores dG_r and dH_r, each reaction data point is numerically labeled for convenience of later reference (e.g. Keq_1, deltaH_23)
        """

        self.all_thermo_data_dict['dG_r'] = {}
        self.all_thermo_data_dict['dH_r'] = {}
        rxns_thermo_data = pd.read_csv(file_name)
        for i, rxn_data_dict in rxns_thermo_data.iterrows():
            cur_r_dict = {'pH': rxn_data_dict['pH'], 'IS': rxn_data_dict['IS'], 'T': rxn_data_dict['T'], 'rxn_formula':rxn_data_dict['rxn_formula'], \
                          'rxn_dict': self.parse_formula(rxn_data_dict['rxn_formula']), 'metal ions': {}}
            metal_ion_list = ['Mg', 'Co', 'Na', 'K', 'Mn', 'Zn', 'Li', 'Ca']
            for metal_ion in metal_ion_list:
                if not pd.isnull(rxn_data_dict[metal_ion]):
                    cur_r_dict['metal ions'][metal_ion] = rxn_data_dict[metal_ion]
            if not pd.isnull(rxn_data_dict['Keq']):
                cur_r_dict['Keq'] = rxn_data_dict['Keq']
                self.all_thermo_data_dict['dG_r'][rxn_data_dict['rxn_id']] = cur_r_dict
            if not pd.isnull(rxn_data_dict['deltaH']):
                cur_r_dict['deltaH'] = rxn_data_dict['deltaH']   
                self.all_thermo_data_dict['dH_r'][rxn_data_dict['rxn_id']] = cur_r_dict
    
    def get_thermo_data(self):
        self.get_compound_data('data/organic_cpd_thermo_data.csv')
        self.get_TECRDB_rxn_data('data/TECRDB_rxn_thermo_data.csv')

if __name__ == '__main__':
    td = thermo_data()        