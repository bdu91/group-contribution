# group-contribution

A python package that calculates the standard Gibbs free energy of reaction as a function of temperature. 

### Requirements

* python == 2.7
* numpy >= 1.13.3
* scipy >= 1.0.0
* python-openbabel >= 2.3.2
* ChemAxon's Marvin >= 16.11.14
* rdkit >= 2017.09.1
* scikit-learn >= 0.19.1
* pandas >= 0.21.0

### Optional requirements

* lmfit >= 0.9.2
* sympy >= 1.1.1

For more information regarding the methods, please refer to [our paper](https://www.cell.com/biophysj/fulltext/S0006-3495(18)30524-1) in Biophysical Journal.


## Installation on Ubuntu

1. `sudo apt-get install openbabel python-rdkit librdkit1 rdkit-data`
2. `sudo pip install -U numpy scipy scikit-learn pandas`
3. ChemAxon [Marvin suite](http://www.chemaxon.com/download/marvin-suite/)
   * add `cxcalc` to PATH
   * obtain a free academic license from ChemAxon
4. Optional
   * `sudo pip install -U lmfit sympy`

## Installation on Windows

1. Install 32-bit python2.7
   * [Winpython](https://github.com/winpython/winpython)
   * add `python.exe` to environment variables
   * for packages not included in WinPython, do `pip install [packagename]`
2. Install openbabel (version 2.3.2) and python bindings (version 1.8)
   * [openbabel](http://openbabel.org/wiki/Category:Installation)
   * [python bindings](http://open-babel.readthedocs.io/en/latest/UseTheLibrary/PythonInstall.html#windows)
3. ChemAxon [Marvin suite](http://www.chemaxon.com/download/marvin-suite/)
   * cxcalc module is typically added to PATH by the installer, if not, depending on where the specific package (e.g. MarvinBeans) of ChemAxon is installed, add `ChemAxon\MarvinBeans\bin` to environment variables
   * obtain a free academic license from ChemAxon
4. Optional
   * installation of [GNU Octave](https://blink1073.github.io/oct2py/source/installation.html)
   * for packages not included in WinPython, do `pip install [packagename]`
   
## Overview of files in /data
* TECRDB_compounds_data.csv, table of all compounds involved in reactions in NIST TECRdb, with their properties including number of hydrogens, charge, binding constant, etc.
* TECRDB_rxn_thermo_data.csv, table of all reaction measurements in NIST TECRdb, with the specific conditions including temperature, ionic strength, pH, metal concentrations, media conditions, equilibrium constants/standard transformed enthalpies of reaction, etc.
* cid_names.csv, compound id and their corresponding names for compounds in TECRdb
* dSf_pKMg_data.csv, table of compounds with standard entropy change of formation or magnesium binding constant data available, as well as molecular descriptors for them and other compounds.
* dSr_training_data.csv, standard entropy change of reaction used as training data for prediction
* metal_thermo_data.csv, thermodynamic properties of metal ions used to calculate the change of standard Gibbs energy of formation of metal ions over temperature
* organic_cpd_thermo_data.csv, thermodynamic properties of collected organic compounds. The specific properties include heat capacity, standard Gibbs free energy of formation, standard transformed Gibbs free energy of formation, standard enthalpy of formation, standard entropy change of formation.
