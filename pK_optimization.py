from lmfit import Parameters, Minimizer
import numpy as np
import re
from time import time
from training_data import training_data
from thermodynamic_transformations import thermodynamic_transformations
R = 8.3144621 #J/K/mol

class pK_optimization(training_data):
    """
    A module to optimize the pK (dissociation constant, can be pKa or pKMg) value of the compound using Keq data.
    The workflow is as follows: 1) find a set of Keq data measured under different pH and Mg concentrations for a
    particular reaction; 2) identify the set of pKs of the metabolites participating in the reaction to be optimized;
    3) write the dGrs of the reaction corrected to standard state for each Keq data, with pKs to be optimized as
    variables; 4) Run least-squares optimization to solve for pKs such that the difference between standard state dGrs
    is minimized
    """
    def __init__(self):
        training_data.__init__(self)
        self.thermo_transform = thermodynamic_transformations(self.TECRDB_compounds_data_dict, self.TECRDB_compounds_pH7_species_id_dict)
        self.get_TECRDB_dSr_data()
        # metabolites that are most frequently found in Keq data from TECRDB and their species ids have pK values that
        # are closet to common pH and Mg concentrations ranges
        self.common_sids2write_sym = ['CHB_15422_-3', 'CHB_15422_-3_Mg1_L1', 'CHB_26078_-1', 'CHB_15422_0_Mg1_L1', \
                                      'CHB_26078_-1_Mg1_L1', 'CHB_16761_-3_Mg2_L1', 'CHB_15422_-1_Mg1_L1', 'CHB_16027_-3_Mg1_L1', \
                                      'CHB_17835_-2', 'CHB_17794_-2', 'CHB_15422_-2_Mg1_L1', 'CHB_16761_-1_Mg1_L1', \
                                      'CHB_16027_-2_Mg1_L1', 'CHB_18009_-2', 'CHB_16474_-3', 'CHB_18361_-3_Mg1_L1', \
                                      'CHB_16761_-3_Mg1_L1', 'CHB_26078_-3_Mg1_L1', 'CHB_15422_-4_Mg2_L1', 'CHB_18361_-3', \
                                      'CHB_18361_-2', 'CHB_18361_-2_Mg1_L1', 'CHB_16761_-2_Mg1_L1', 'CHB_16761_-2', \
                                      'CHB_16027_-1', 'CHB_26078_-2_Mg1_L1']
    
    def setup_optm_params(self, pK_var_range=0.5,xtol_value=1e-10,ftol_value=1e-05,gtol_value=1e-10,maxfev_value=100000,epsfcn_min_value = 6):
        """
        the optimization parameters to be used in later least-square optimization
        """
        self.pK_var_range = pK_var_range
        self.xtol_value = xtol_value
        self.ftol_value = ftol_value
        self.gtol_value = gtol_value
        self.maxfev_value = maxfev_value
        self.epsfcn_min_value = epsfcn_min_value
        
    def write_dGr0_equations(self, rid_list):
        """
        Generate dGr0 equations with pK to be optimized as variables
        :param rid_list: a list of reaction ids
        :return: a dictionary with key being reaction id, value being equation for dGr0 of the reaction
        """
        dGr0_equations_dict = {}
        print 'Writing %d dGr0 equations' %len(rid_list)
        for rid in rid_list:
            dGr0_equations_dict[rid] = self.thermo_transform._get_dGr0_sym(self.all_thermo_data_dict['dG_r'], \
                                       self.TECRDB_rxn_dSr_dict, rid, metal_correction=True)
        return dGr0_equations_dict
    
    @staticmethod
    def merge_list_of_list(list_of_list):
        """
        :param list_of_list: a list of sublist
        :return: single list containing items from all sublists
        """
        return [item for cur_list in list_of_list for item in cur_list]
    
    @staticmethod
    def extract_variable_index_from_string(string):
        """
        find the index from a symbol written as the variable, e.g. return 0 for x[0], 2 for x[2]
        """
        indices = map(int, re.findall('(?<=x\[)([0-9]+)(?=\])',string))
        return indices[0]
    
    @staticmethod
    def combine_similar_vals(val_list, threshold=0.1):
        '''
        group similar values, mainly used group measurements with close pKMg (here default to be 0.1)
        '''
        similar_val_dict = {}
        cur_dict_key = 0
        for i, cur_val in enumerate(val_list):
            if similar_val_dict == {}:
                similar_val_dict[cur_dict_key] = {'mean': cur_val, 'indices':[i],'values':[cur_val]}
            else:
                val_assigned = 0
                for data_dict in similar_val_dict.values():
                    if abs(data_dict['mean'] - cur_val) <= threshold and val_assigned == 0:
                        data_dict['values'].append(cur_val)
                        data_dict['indices'].append(i)
                        data_dict['mean'] = np.mean(data_dict['values'])
                        val_assigned = 1
                if val_assigned == 0:
                    cur_dict_key += 1
                    similar_val_dict[cur_dict_key] = {'mean': cur_val, 'indices':[i], 'values':[cur_val]}
        return similar_val_dict
    
    def residual_func_euclid(self, params):
        """
        Function that calculates the residuals between the actual data values
        and the predicted values (i.e. 'Data - Model'). Returns a numpy array.
        """
        newC = []  # Same as candidate outside of this function
        for name in params:
            newC.append(params[name].value)
        newC = np.array(newC)

        dGr0_val = []
        for rid in self.rids2optm:
            dGr0_val.append(self.functionDict[rid](newC))
        dGr0_val = np.array(dGr0_val)
        
        sqrd_error_list = []
        start = 0
        for i, cur_rid_list in enumerate(self.rids2optm_rxn_list):
            #caculate the sum of squared error for each reaction and then multiply by the weights
            cur_dGr0_val_list = dGr0_val[start:start + len(cur_rid_list)]
            cur_residual = cur_dGr0_val_list - np.mean(cur_dGr0_val_list)
            sqrd_error_list.append(sum(np.power(cur_residual,2)) * self.weights[i])
            start += len(cur_rid_list)
            
        sse = sum(sqrd_error_list)
        return np.array([sse]*len(params)) #for optimization, the length of objective value should be at least the length of parameters
    
    def _minimizerApprox(self, candidate, methodApprox="lbfgsb",tolApprox=10**-3):
        '''
        Initial minimizer before applying leastsq fit, helps faster convergence
        '''
        #Create a Parameter Object
        params = Parameters()
        for index in range(0,len(candidate)):
            paramName = 'c%s' % index   # Create unique names for each parameter
            params.add(paramName, value= candidate[index], min= self.minCandidates[index], max= self.maxCandidates[index])

        # Actual Minimization
        fitter = Minimizer(self.residual_func_euclid, params)
        resultApprox = fitter.minimize(method = methodApprox, tol = tolApprox)
        return resultApprox
    
    def _minimizerLMA(self, result):
        """
        Lower level wrapper function for running 'lmfit.minimize()'. This function
        accepts a candidate set and returns the minimized candidate set. A course to 
        fine approach is used to set the suitable step length for the forward-difference
        approximation of the Jacobian (epsfcn).
        """
        #Minimize using a decreasing gradient approximation stepsize and an euclidean space objective function
        for i in range(0, self.epsfcn_min_value + 1):
            e_value = 10.0**(-i)
            # Actual Minimization
            fitter = Minimizer(self.residual_func_euclid,result.params)
            result = fitter.minimize(method='leastsq', params=result.params, xtol=self.xtol_value, ftol=self.ftol_value, gtol=self.gtol_value, epsfcn=e_value, maxfev=self.maxfev_value)
        return result
    
    def setup_vars_and_eqns(self, rids2optm_rxn_list, rids2write_eqns, sids2write_sym, weights = []):
        """
        Set up variables and dG0 equations containing variables
        :param rids2optm_rxn_list: a list of list, where each sublist contains the reaction ids of the same reaction whose dGr0s are used to optimize pK
        :param rids2write_eqns: reaction ids to write dG0 equations, contain all rids from rids2optm_rxn_list and other reaction ids that will use the optimized pK
        :param sids2write_sym: species ids whose pKs are to write as variables
        :param weights: weightings for the reactions used to optimize pK values, length equals the length of rids2optm_rxn_list, that is the number of different reactions
        """
        self.rids2optm_rxn_list = rids2optm_rxn_list
        self.rids2optm = pK_optimization.merge_list_of_list(self.rids2optm_rxn_list)
        self.rids2write_eqns = rids2write_eqns
        self.sids2write_sym = sids2write_sym
        if weights == []:
            self.weights = np.full(len(self.rids2optm_rxn_list), 1.0)
        else:
            self.weights = weights
        
        #Get species ids whose pKs are of interest to write symbols and participate in reactions that will be written into dGr_standard equations
        cids_in_rids2optm = list(set(pK_optimization.merge_list_of_list([self.all_thermo_data_dict['dG_r'][rid]['rxn_dict'].keys() for rid in self.rids2optm])))
        self.selected_sids2write_sym = [sid for sid in self.sids2write_sym if self.TECRDB_compounds_data_dict[sid]['compound_id'] in cids_in_rids2optm]
        self.thermo_transform.setup_symbols_for_species_pKs(self.selected_sids2write_sym)
        
        #Now set up dGr0_equations and corresponding functions
        self.dGr0_eqns_dict = self.write_dGr0_equations(self.rids2write_eqns)
        self.functionDict = {}
        self.dGr0_val_no_Mg_corr = {} #also calculate dGr0 without metal correction
        for rid in self.rids2write_eqns:
            self.dGr0_val_no_Mg_corr[rid] = self.thermo_transform._get_dGr0_num(self.all_thermo_data_dict['dG_r'], \
                                           self.TECRDB_rxn_dSr_dict, rid)
            cur_dGr0_eqn = self.dGr0_eqns_dict[rid]
            func_name = 'f_' + rid
            make_func_command = 'def %s(x): return %s' % (func_name, cur_dGr0_eqn)
            exec(make_func_command)
            self.functionDict[rid] = eval(func_name)
        
        #Also set up the vector that contains the original pK value
        self.var_vector_to_use = np.zeros(len(self.thermo_transform.variable_vector_dict))
        for pK_variable_info in self.thermo_transform.variable_vector_dict.values():
            pK_symbol = pK_variable_info[0]; pK_val = pK_variable_info[1]
            index = pK_optimization.extract_variable_index_from_string(str(pK_symbol))
            self.var_vector_to_use[index] = pK_val

        
    def run_optimization(self, candidates_provided = []):
        """
        LMA optimization to minimize dGr0 variation using pK as variable
        """
        #setup the candidates that go into the minimizer
        self.minCandidates = [var - self.pK_var_range for var in self.var_vector_to_use]
        self.maxCandidates = [var + self.pK_var_range for var in self.var_vector_to_use]

        if candidates_provided == []:
            #if no candidates are provided, set up initialized random starting points
            self.candidates = [np.random.uniform(mincand,self.maxCandidates[i],1)[0] if str(mincand) != 'nan' else np.nan for i, mincand in enumerate(self.minCandidates)]
        else:
            #we use candidates_provided if it is non-empty
            self.candidates = candidates_provided

        #first run approx optimization
        start_timer_approx = time()
        self.approx_results = self._minimizerApprox(self.candidates)
        end_timer_approx = time()
        print 'Elapsed time %f seconds for approximate optimization' %round(end_timer_approx-start_timer_approx, 3)
        
        #Now run lma optimization
        start_timer_lma = time()
        self.leastsq_results = self._minimizerLMA(self.approx_results)
        end_timer_lma = time()
        print 'Elapsed time %f seconds for LMA optimization' %round(end_timer_lma-start_timer_lma, 3)
        
        self.minimized_candidate = []
        for name in self.leastsq_results.params: 
            self.minimized_candidate.append(self.leastsq_results.params[name].value)
        self.minimized_candidate = np.array(self.minimized_candidate)
        
    def write_optm_pKs_for_sids(self):
        """
        Export optimized pK value for species id
        :return: a dictionary with key being species id and value being optimized pK
        """
        optimized_sid_pK_dict = {}
        for sid, pK_variable_info in self.thermo_transform.variable_vector_dict.iteritems():
            pK_symbol = pK_variable_info[0]
            index = pK_optimization.extract_variable_index_from_string(str(pK_symbol))
            optimized_sid_pK_dict[sid] = self.minimized_candidate[index]
        return optimized_sid_pK_dict
                    
    def compare_optm_results(self,rids2compare, initial_candidates = [], final_candidates = []):
        """
        Set up dGr0 without Mg correction, with Mg correction using original Mg values, using optimized values
        """
        #rids_to_compare need to include reaction id that have been written into equations
        self.rids2compare = rids2compare
        #we want to examine different initial and optimized endpoints directly if we have final minimized candidates available
        if initial_candidates == []:
            self.dGr0_w_original_val = [self.functionDict[rid](self.var_vector_to_use) for rid in self.rids2compare]
        else:
            self.dGr0_w_original_val = [self.functionDict[rid](initial_candidates) for rid in self.rids2compare]

        if final_candidates == []:
            self.dGr0_w_optimized_val = [self.functionDict[rid](self.minimized_candidate) for rid in self.rids2compare]
        else:
            self.dGr0_w_optimized_val = [self.functionDict[rid](final_candidates) for rid in self.rids2compare]
        
        self.pMg_to_plot = [-np.log10(self.all_thermo_data_dict['dG_r'][rid]['metal ions']['Mg']) for rid in self.rids2compare]
        pMg_similar_vals_dict = pK_optimization.combine_similar_vals(self.pMg_to_plot)
        
        self.dGr_prime_list = [-R*self.all_thermo_data_dict['dG_r'][rid]['T']*np.log(self.all_thermo_data_dict['dG_r'][rid]['Keq']) for rid in self.rids2compare]
        self.dGr0_no_Mg_corr = [self.dGr0_val_no_Mg_corr[rid] for rid in self.rids2compare]
        
        self.updated_pMg2plot = []
        self.updated_dGr0_w_original_val = []
        self.updated_dGr0_w_optimized_val = []
        self.updated_dGr0_no_Mg_corr = []
        self.updated_dGr_prime_list = []
        for data_dict in pMg_similar_vals_dict.values():
            self.updated_pMg2plot.append(data_dict['mean'])
            self.updated_dGr_prime_list.append(np.mean([self.dGr_prime_list[cur_index] for cur_index in data_dict['indices']]))
            self.updated_dGr0_w_original_val.append(np.mean([self.dGr0_w_original_val[cur_index] for cur_index in data_dict['indices']]))
            self.updated_dGr0_w_optimized_val.append(np.mean([self.dGr0_w_optimized_val[cur_index] for cur_index in data_dict['indices']]))
            self.updated_dGr0_no_Mg_corr.append(np.mean([self.dGr0_no_Mg_corr[cur_index] for cur_index in data_dict['indices']]))