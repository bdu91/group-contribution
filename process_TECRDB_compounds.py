import pandas as pd

class TECRDB_compounds_data(object):
    def __init__(self):
        """
        A module that processes information of compounds in TECRDB
        """
        self.TECRDB_compounds_data_dict = {}
        self.TECRDB_compounds_pH7_species_id_dict = {}
        self.TECRDB_compounds_least_H_sid_dict = {}
        self.get_TECRDB_compounds_data()
        
    def get_TECRDB_compounds_data(self):
        """
        reads in data for compounds in TECRDB
        :return: a dictionary with keys being different ion bound states of the compound (we call it species_id here, e.g CHB_15422_-1 refers to -1 charged form of
        compound_id CHB_15422), values being a dictionary storing the thermodynamic information and molecular properties of the species_id
        """
        TECRDB_compounds_data_table = pd.read_csv('data/TECRDB_compounds_data.csv')
        #all possible information that the particular ion bound state can have
        data_entry_list = ['Cp', 'H_number', 'binding_constant', 'charge', 'dG_f', 'dH_f', 'dS_f', 'groups', 'metal_type','smiles_form','metal_number']
        for i, row in TECRDB_compounds_data_table.iterrows():
            cur_sid = row['species_id']
            cur_cid = row['compound_id']
            self.TECRDB_compounds_data_dict[cur_sid] = {'compound_id':cur_cid}
            if row['is_pH7_species'] == True:
                self.TECRDB_compounds_pH7_species_id_dict[cur_cid] = cur_sid
            if row['is_least_protonated_species'] == True:
                self.TECRDB_compounds_least_H_sid_dict[cur_cid] = cur_sid
            for data_entry in data_entry_list:
                if not pd.isnull(row[data_entry]):
                    if data_entry == 'groups':
                        #convert the text form of groups to python list
                        cur_sid_groups = map(float,row['groups'].strip('[').strip(']').split(','))
                        self.TECRDB_compounds_data_dict[cur_sid]['groups'] = cur_sid_groups
                    else:
                        try:
                            #convert value from string to float
                            self.TECRDB_compounds_data_dict[cur_sid][data_entry] = float(row[data_entry])
                        except ValueError:
                            self.TECRDB_compounds_data_dict[cur_sid][data_entry] = row[data_entry]
                            
if __name__ == '__main__':
    TECRDB_cpd_data = TECRDB_compounds_data()