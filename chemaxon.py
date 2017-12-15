from subprocess import Popen, PIPE
import openbabel
import logging
import numpy as np
import StringIO
import re, csv
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
#from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors #the list below defines all molecular descriptors available

class ChemAxonError(Exception):
    pass

def RunCxcalc(molstring, args):
    """
    Call cxcalc argument of ChemAxon from command line
    :param molstring: smiles form or InChI strings of the molecule
    :param args: name of molecular descriptors processed by cxcalc
    :return: output from command line cxcalc requring further processing
    """
    CXCALC_BIN = "cxcalc"
    devnull = open('/dev/null', 'w')
    try:
        p1 = Popen(["echo", molstring], stdout=PIPE)
        p2 = Popen([CXCALC_BIN] + args, stdin=p1.stdout,
                   executable=CXCALC_BIN, stdout=PIPE, stderr=devnull)
        logging.debug("INPUT: echo %s | %s" % (molstring, ' '.join([CXCALC_BIN] + args)))
        #p.wait()
        #os.remove(temp_fname)
        res = p2.communicate()[0]
        if p2.returncode != 0:
            raise ChemAxonError(debug_args)
        logging.debug("OUTPUT: %s" % res)
        return res
    except OSError:
        raise Exception("Marvin (by ChemAxon) must be installed to calculate pKa data.")

def Molconvert(molstring, args):
    """
    Call molconvert argument of ChemAxon from command line
    :param molstring: smiles form or InChI strings of the molecule
    :param args: name of molecular descriptors processed by molconvert
    :return: output from command line molconvert requring further processing
    """
    MOLCONV_BIN = "molconvert"
    devnull = open('/dev/null', 'w')
    try:
        p1 = Popen(["echo", molstring], stdout=PIPE)
        p2 = Popen([MOLCONV_BIN] + args, stdin=p1.stdout,
                   executable=MOLCONV_BIN, stdout=PIPE, stderr=devnull)
        logging.debug("INPUT: echo %s | %s" % (molstring, ' '.join([MOLCONV_BIN] + args)))
        #p.wait()
        #os.remove(temp_fname)
        res = p2.communicate()[0]
        if p2.returncode != 0:
            raise ChemAxonError(debug_args)
        logging.debug("OUTPUT: %s" % res)
        return res
    except OSError:
        raise Exception("Marvin (by ChemAxon) must be installed")

def inchi2smiles(inchi):
    openbabel.obErrorLog.SetOutputLevel(-1)

    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats('inchi', 'smiles')
    # conv.AddOption("F", conv.OUTOPTIONS)
    # conv.AddOption("T", conv.OUTOPTIONS)
    # conv.AddOption("x", conv.OUTOPTIONS, "noiso")
    # conv.AddOption("w", conv.OUTOPTIONS)
    obmol = openbabel.OBMol()
    conv.ReadString(obmol, str(inchi))
    smiles = conv.WriteString(obmol, True)  # second argument is trimWhitespace
    if smiles == '':
        return None
    else:
        return smiles

def smiles2inchi(smiles):
    openbabel.obErrorLog.SetOutputLevel(-1)

    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats('smiles', 'inchi')
    conv.AddOption("F", conv.OUTOPTIONS)
    conv.AddOption("T", conv.OUTOPTIONS)
    conv.AddOption("x", conv.OUTOPTIONS, "noiso")
    conv.AddOption("w", conv.OUTOPTIONS)
    obmol = openbabel.OBMol()
    conv.ReadString(obmol, str(smiles))
    inchi = conv.WriteString(obmol, True)  # second argument is trimWhitespace
    if inchi == '':
        return None
    else:
        return inchi
        
def GetFormulaAndCharge(molstring):
    """
    :param molstring: smiles form or InChI strings of the molecule
    :return: chemical formula and charge of the molecule
    """
    args = ['formula', 'formalcharge']
    output = RunCxcalc(molstring, args)
    # the output is a tab separated table whose columns are:
    # id, Formula, Formal charge
    f = StringIO.StringIO(output)
    tsv_output = csv.reader(f, delimiter='\t')
    headers = tsv_output.next()
    if headers != ['id', 'Formula', 'Formal charge']:
        raise ChemAxonError('cannot get the formula and charge for: ' + molstring)
    _, formula, formal_charge = tsv_output.next()

    try:
        formal_charge = int(formal_charge)
    except ValueError:
        formal_charge = 0
    
    return formula, formal_charge
        
def GetAtomBagAndCharge(molstring):
    """
    :param molstring: smiles form or InChI strings of the molecule
    :return: - a dictionary with key being atom value being number of atoms in the molecule
             - charge of the molecule
    """
    formula, formal_charge = GetFormulaAndCharge(molstring)

    atom_bag = {}
    for mol_formula_times in formula.split('.'):
        for times, mol_formula in re.findall('^(\d+)?(\w+)', mol_formula_times):
            if not times:
                times = 1
            else:
                times = int(times)
            for atom, count in re.findall("([A-Z][a-z]*)([0-9]*)", mol_formula):
                if count == '':
                    count = 1
                else:
                    count = int(count)
                atom_bag[atom] = atom_bag.get(atom, 0) + count * times

    return atom_bag, formal_charge

def _GetDissociationConstants(molstring, n_acidic=20, n_basic=20, pH=7.0):
    """
    :param molstring: smiles form or InChI strings of the molecule
    :return: A pair of (pKa list, major pseudoisomer)
             - pKa list is of a list of pKa values in ascending order.
             - the major pseudoisomer is a SMILES string of the major species at the given pH.
    """
    args = []
    if n_acidic + n_basic > 0:
        args += ['pka', '-a', str(n_acidic), '-b', str(n_basic),
                 'majorms', '-M', 'true', '--pH', str(pH)]

    output = RunCxcalc(molstring, args)
    atom2pKa, smiles_list = ParsePkaOutput(output, n_acidic, n_basic)

    all_pKas = []
    for pKa_list in atom2pKa.values():
        all_pKas += [pKa for pKa, _ in pKa_list]

    return sorted(all_pKas), smiles_list

def ParsePkaOutput(s, n_acidic, n_basic):
    """
    :param s: output of pKa values
    :return:  A dictionary that maps the atom index to a list of pKas that are assigned to that atom.
    """
    atom2pKa = {}

    pkaline = s.split('\n')[1]
    splitline = pkaline.split('\t')
    splitline.pop(0)

    if n_acidic + n_basic > 0:
        if len(splitline) != (n_acidic + n_basic + 2):
            raise ChemAxonError('ChemAxon failed to find any pKas')

        pKa_list = []
        acid_or_base_list = []
        for i in range(n_acidic + n_basic):
            x = splitline.pop(0) 
            if x == '':
                continue

            pKa_list.append(float(x))
            if i < n_acidic:
                acid_or_base_list.append('acid')
            else:
                acid_or_base_list.append('base')

        atom_list = splitline.pop(0)

        if atom_list: # a comma separated list of the deprotonated atoms
            atom_numbers = [int(x)-1 for x in atom_list.split(',')]
            for i, j in enumerate(atom_numbers):
                atom2pKa.setdefault(j, [])
                atom2pKa[j].append((pKa_list[i], acid_or_base_list[i]))

    smiles_list = splitline
    return atom2pKa, smiles_list

def GetDissociationConstants(molstring, n_acidic=20, n_basic=20, pH=7):
    """
    Get pKas and major microspecies of the molecule
    :param molstring: smiles form or InChI strings of the molecule
    :param n_acidic: the max no. of acidic pKas to calculate
    :param n_basic: the max no. of basic pKas to calculate
    :param pH: the pH for which the major pseudoisomer is calculated
    :return: (all_pKas, major_ms)
            - all_pKas is a list of floats (pKa values)
            - major_ms is a SMILES string of the major pseudoisomer at pH_mid
    """

    all_pKas, smiles_list = _GetDissociationConstants(molstring, n_acidic, 
                                                      n_basic, pH)
    major_ms = smiles_list[0]
    pKas = sorted([pka for pka in all_pKas if pka > 0 and pka < 13], reverse=True)
    return pKas, major_ms

def Get_pKas_Hs_zs_pH7smiles(molstring):
    """
    :param molstring: smiles form or InChI strings of the molecule
    :return: a list of pKas, a list of H atom number for each protonation state, a list of charges, index of major species at pH 7, the smiles form of the major
    species at pH 7
    """

    pKas, major_ms_smiles = GetDissociationConstants(molstring)
    pKas = sorted([pka for pka in pKas if pka > 0 and pka < 13], reverse=True)
    #print major_ms_smiles

    if major_ms_smiles:
        atom_bag, major_ms_charge = GetAtomBagAndCharge(major_ms_smiles)
        major_ms_nH = atom_bag.get('H', 0)
    else:
        atom_bag = {}
        major_ms_charge = 0
        major_ms_nH = 0

    n_species = len(pKas) + 1
    if pKas == []:
        majorMSpH7 = 0
    else:
        majorMSpH7 = len([1 for pka in pKas if pka > 7])

    nHs = []
    zs = []

    for i in xrange(n_species):
        zs.append((i - majorMSpH7) + major_ms_charge)
        nHs.append((i - majorMSpH7) + major_ms_nH)

    return pKas, nHs, zs, majorMSpH7, major_ms_smiles

def Calculate_total_Steric_hindrance(molstring):
    """
    :param molstring: smiles form or InChI strings of the molecule
    :return: total steric hindrance of the molecule
    """
    # convert to smiles form if it is InChI string
    if "InChI=" in molstring:
        molstring = inchi2smiles(molstring)

    molstring_with_explicit_Hs = Molconvert(molstring, ['smiles:H']).split('\n')[0]
    steric_hindrance = sorted(map(float, RunCxcalc(molstring_with_explicit_Hs,['sterichindrance','-l','always']).split('\n')[1].split('\t')[1].split(';')))
    return sum(steric_hindrance)

def Find_pos_of_double_bond_O(molstring):
    """
    :param molstring: smiles form or InChI string of the molecule
    :return: position of double bond oxygen atom in the list of atoms in molstring
    """
    #convert to smiles form if it is InChI string
    if "InChI=" in molstring:
        molstring = inchi2smiles(molstring)

    atoms_to_consider = ['C','O','N','S','P','c','n','Cl']
    double_bond_O_pos = []
    if 'O=C' in molstring:
        double_bond_O_pos.append(0)
    if 'O=c' in molstring:
        double_bond_O_pos.append(0)

    molstring_split_by_double_bond_O = molstring.split('=O')
    
    if len(molstring_split_by_double_bond_O) > 1:
        atom_num_in_fragments = []
        for smiles_fragment in molstring_split_by_double_bond_O:
            atom_num_in_fragments.append(sum([smiles_fragment.count(cur_atom) for cur_atom in atoms_to_consider]))
        double_bond_O_pos_real = np.cumsum([atom_num + 1 for atom_num in atom_num_in_fragments])[:-1] #not counting from 0
        double_bond_O_python_pos = list(np.array(double_bond_O_pos_real) - 1)
    else:
        double_bond_O_python_pos = []
    
    double_bond_O_pos += double_bond_O_python_pos
    
    return double_bond_O_pos

def Extract_individual_atom_partial_charge_and_labels(molstring):
    """
    :param molstring: smiles form or InChI string of the molecule
    :return: (atom_labels, atom_partial_charge)
            - atom_labels is a list of atoms in the molecule specified by the their atom type
            - atom_partial_charge is a list of partial charge for each atom in the molecule
    """
    #convert to smiles form if it is InChI string
    if "InChI=" in molstring:
        molstring = inchi2smiles(molstring)
    partial_charge_output = RunCxcalc(molstring, ['-M','charge','-i','True','-p','3'])
    #depending on smiles, the partial charge output from chemaxon can have two different formats
    if len(partial_charge_output.split('</atom>')) > 1:
        #one particular output format
        atom_labels = [re.findall('elementType=(.*)',element)[0].strip('"')[0] for element in partial_charge_output.split('</atom>') if 'elementType' in element]
        atom_partial_charge = [re.findall('mrvExtraLabel=(.*) x2',element)[0].strip('"') for element in partial_charge_output.split('</atom>') if 'elementType' in element]
    else:
        #another particular output format
        if 'formalCharge' in partial_charge_output:
            atom_labels = re.findall('elementType=(.*) formalCharge',partial_charge_output)[0].strip('"').split(' ')
        else:
            atom_labels = re.findall('elementType=(.*) mrvExtraLabel',partial_charge_output)[0].strip('"').split(' ')
        atom_partial_charge = re.findall('mrvExtraLabel=(.*) x2=',partial_charge_output)[0].strip('"').split(' ')
    return atom_labels, atom_partial_charge

def Extract_atom_partial_charge(molstring, absolute_charge = True):
    """
    Extract the total absolute partial charge for each type of atom
    :param molstring: smiles form or InChI string of the molecule
    :param absolute_charge: if we take the absolute value for the partial charge of each atom when calculating the total partial charge
    :return: a dictionary with keys being atom type, values being total absolute partial charge of the type of atom
    """
    #convert to smiles form if it is InChI string
    if "InChI=" in molstring:
        molstring = inchi2smiles(molstring)
    atom_charge_dict = {'C':[],'H':[],'O':[],'O_double':[],'N':[],'S':[],'P':[],'Cl':[],'F':[],'Br':[],'I':[]} #Br and I as placeholder, currently don't have data to predict compounds containing these elements
    total_charge_dict = {}
    atom_labels, atom_partial_charge = Extract_individual_atom_partial_charge_and_labels(molstring)
    double_bond_O_pos = Find_pos_of_double_bond_O(molstring)
    H_charge_list = [float(re.findall('\((.*)\)',partial_charge)[0]) for partial_charge in atom_partial_charge if '(' in partial_charge]
    #atom_labels correspond to all_other_atoms_charge_list
    all_other_atoms_charge_list = [float(partial_charge.split('\\')[0]) for partial_charge in atom_partial_charge]    
    
    for i, atom_type in enumerate(atom_labels):
        if atom_type == 'O':
            if i in double_bond_O_pos:
                atom_charge_dict['O_double'].append(all_other_atoms_charge_list[i])
            else:
                atom_charge_dict['O'].append(all_other_atoms_charge_list[i])        
        else:
            atom_charge_dict[atom_type].append(all_other_atoms_charge_list[i])
    
    if H_charge_list != []:
        atom_charge_dict['H'] += H_charge_list
    
    for atom_type, charge_list in atom_charge_dict.iteritems():
        if charge_list != []:
            if absolute_charge == True:
                total_charge_dict[atom_type] = np.sum(abs(np.array(charge_list)))
            else:
                total_charge_dict[atom_type] = np.sum(np.array(charge_list))
    return total_charge_dict

def Calculate_total_atom_num_and_partial_charge(molstring):
    """
    :param molstring: smiles form or InChI string of the molecule
    :return: a list of total atom number and absolute partial charge for each atom type
            [C_partial_charge, C_atom_num, H_partial_charge, H_atom_num, O_partial_charge, O_atom_num, O_double_partial_charge, O_double_atom_num,
            N_partial_charge, N_atom_num, S_partial_charge, S_atom_num, P_partial_charge, P_atom_num, F_partial_charge, F_atom_num, Cl_partial_charge, Cl_atom_num]
    """
    cur_total_atom_num_partial_charge_array = []
    atom_bag, _ = GetAtomBagAndCharge(molstring)
    double_bond_O_num = len(Find_pos_of_double_bond_O(molstring))
    if 'O' in atom_bag.keys():
        single_bond_O_num = atom_bag['O'] - double_bond_O_num
        atom_bag['O'] = single_bond_O_num
        atom_bag['O_double'] = double_bond_O_num

    partial_charge_dict = Extract_atom_partial_charge(molstring)

    for atom_type in ['C','H','O','O_double','N','S','P','F','Cl']:
        if atom_type in partial_charge_dict.keys():
            cur_total_atom_num_partial_charge_array.append(partial_charge_dict[atom_type])
        else:
            cur_total_atom_num_partial_charge_array.append(0.0)

        if atom_type in atom_bag.keys():
            cur_total_atom_num_partial_charge_array.append(atom_bag[atom_type])
        else:
            cur_total_atom_num_partial_charge_array.append(0.0)
    #total atomcount covered in molecular properties calculation
    #cur_total_atom_num_partial_charge_array.append(sum(atom_bag.values()))
    return cur_total_atom_num_partial_charge_array

def Calculate_chemaxon_mol_properties(molstring):
    """
    Calculate the molecular descriptors available in ChemAxon
    :param molstring: smiles form or InChI string of the molecule
    :return: a list of ChemAxon molecular descriptors of the molecule
    """
    args = ['atomcount','exactmass','averagemolecularpolarizability','axxpol','ayypol','azzpol','formalcharge',\
            'molecularpolarizability','aliphaticatomcount','aliphaticbondcount','aliphaticringcount','aromaticatomcount',\
            'aromaticbondcount','aromaticringcount','asymmetricatomcount','balabanindex','bondcount','carboaromaticringcount',\
            'carboringcount','chainatomcount','chainbondcount','chiralcentercount','cyclomaticnumber','dreidingenergy',\
            'fusedaromaticringcount','fusedringcount','hararyindex','heteroaliphaticringcount','heteroaromaticringcount',\
            'heteroringcount','hyperwienerindex','largestringsize','largestringsystemsize','maximalprojectionarea',\
            'maximalprojectionradius','maximalprojectionsize','minimalprojectionarea','minimalprojectionradius',\
            'minimalprojectionsize','mmff94energy','molecularsurfacearea','plattindex','psa','randicindex','ringatomcount',\
            'ringbondcount','ringcount','ringsystemcount','rotatablebondcount','smallestringsize','smallestringsystemsize',\
            'stereodoublebondcount','szegedindex','volume','wienerindex','wienerpolarity','tautomercount','logp',\
            'acceptorcount','acceptorsitecount','donorcount','donorsitecount','refractivity','resonantcount','asa','dipole']
    chemaxon_output = RunCxcalc(molstring, args)
    mol_property_names = chemaxon_output.split('\n')[0].split('\t')[1:]
    mol_property_vals = chemaxon_output.split('\n')[1].split('\t')[1:]
    
    try:
        averagemicrospeciescharge = RunCxcalc(molstring, ['averagemicrospeciescharge']).split('\n')[1].split('\t')[2]
    except IndexError:
        averagemicrospeciescharge = RunCxcalc(molstring, ['formalcharge']).split('\n')[1].split('\t')[1]
    mol_property_names.append('averagemicrospeciescharge')
    mol_property_vals.append(averagemicrospeciescharge)
    
    #both related to issues where logp of smiles cannot be calculated, such as H2 ([H][H]), raise a warning message regarding this issue
    property_error_count = 0
    for i, cur_property_val in enumerate(mol_property_vals):
        if cur_property_val == 'logp:FAILED' or cur_property_val == '':
            mol_property_vals[i] = '0'
            property_error_count += 1
    if property_error_count > 0:
        print 'Problem calculating logp for %s, the molecular properties calculated might not be correct' %molstring

    mol_property_vals = map(float, mol_property_vals)
    double_bond = float(molstring.count('=')); mol_property_vals.append(double_bond); mol_property_names.append('double_bond_count')
    triple_bond = float(molstring.count('#')); mol_property_vals.append(triple_bond); mol_property_names.append('triple_bond_count') 
    #print mol_property_names
    return mol_property_vals

rdkit_descriptors = ['BalabanJ','BertzCT','FractionCSP3','HallKierAlpha','HeavyAtomCount','HeavyAtomMolWt',\
'Kappa1','Kappa2','Kappa3','LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',\
'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge','MolLogP','MolMR','NHOHCount','NOCount',\
'NumHeteroatoms','NumRotatableBonds','NumValenceElectrons','TPSA']
rdkit_mol_descrip_calculator = MolecularDescriptorCalculator(rdkit_descriptors)

def Calculate_rdkit_mol_descriptors(mol_string):
    """
    Calculate the molecular descriptors available in RDkit
    :param mol_string: smiles form or InChI string of the molecule
    :return: a list of RDkit molecular descriptors of the molecule
    """
    if 'InChI=' in mol_string:
        mol_string = inchi2smiles(mol_string)
    cur_molecule = Chem.MolFromSmiles(mol_string)
    cur_mol_properties = rdkit_mol_descrip_calculator.CalcDescriptors(cur_molecule)
    return list(cur_mol_properties)

def Calculate_mol_properties(mol_string):
    """
    Calculate all molecular descriptors on partial charge, steric hindrance, ChemAxon and RDkit molecular descriptors
    :param mol_string: smiles form or InChI string of the molecule
    :return: the molecular descriptors of the molecule
    """
    if 'InChI=' in mol_string:
        mol_string = inchi2smiles(mol_string)
    total_atom_num_and_partial_charge = Calculate_total_atom_num_and_partial_charge(mol_string)
    total_steric_hindrance = [Calculate_total_Steric_hindrance(mol_string)]
    chemaxon_mol_properties = Calculate_chemaxon_mol_properties(mol_string)
    rdkit_mol_properties = Calculate_rdkit_mol_descriptors(mol_string)
    all_mol_properties = total_atom_num_and_partial_charge + total_steric_hindrance + chemaxon_mol_properties + rdkit_mol_properties
    return all_mol_properties