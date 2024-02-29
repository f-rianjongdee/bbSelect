#!/usr/bin/env python

"""
Contains the functions to process the bbSelect database to perform a selection

@author: Francesco Rianjongdee
"""

import os
import sys
from importlib.resources import files
import data
import rdkit
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom
from rdkit.Chem import PandasTools
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_3d = True
#print(rdkit.__version__)
import numpy as np
import math
import logging
import time
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from bitarray import bitarray
import itertools
import multiprocessing
import subprocess

# Complex logging to send warnings to warning_logs.log and info to sys.stdout

# Adjusting the custom filter for the console handler
class LessThanWarningFilter(logging.Filter):
    def filter(self, record):
        # Allow log records with level less than WARNING
        return record.levelno < logging.WARNING
    
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Clear existing handlers and reconfigure the root logger
root_logger.handlers.clear()
root_logger.setLevel(logging.DEBUG)  # Set to capture all levels of messages

# Console handler setup with the new filter
console_handler = logging.StreamHandler()
console_handler.addFilter(LessThanWarningFilter())  # Apply the new filter
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# File handler setup remains the same (for WARNING level)
log_text = open('warning_logs.txt', "w")
log_text.close()

file_handler = logging.FileHandler('warning_logs.txt')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

class bbSelectBuild():
    """
    Further simplifies the process to be run in two lines of code.
    """

    def __init__(self, smiles_file, output_root, cell_size, num_cells, ncpu, rxn_smarts):

        self.options = {
        'output_root' : output_root,
        'num_cells'  : num_cells,
        'ncpu' : ncpu,
        'rxn_smarts' : rxn_smarts,
        'smiles_file_sep' : ',',
        'smiles_file' : smiles_file,
        'clipped_smiles_file' : None,
        'clipped_smiles_file_sep' : '\t',
        'aligned_mols_sdf': None,
        'save_aligned_mols_sdf' : None,
        'conformer_sdf' : None,
        'conformer_sdf_save_location' : None,
        'conformer_generation_program' : 'omega',
        'omega_load_command' : 'module load openeye',
        'omega_run_command' : 'omega2',
        'pharmacophore_family_dict' : 'default',
        'cell_size' : cell_size,
        }

    def PrintOptions(self):
        for option in self.options.keys():
            print(f'{option} : {self.options[option]}')

    def SetOption(self, option, value):
        self.options[option] = value

    def Run(self):

        ## Initiate builder
        bbSelectBuilder = bbSelectDBbuilder(cell_size = self.options['cell_size'],
            num_cells = self.options['num_cells'],
            pharmacophore_family_dict = self.options['pharmacophore_family_dict'],
            ncpu = self.options['ncpu'])

        ## Load smiles file
        logging.info('loading smiles file')
        bbSelectBuilder.load_smiles_file(self.options['smiles_file'], sep = self.options['smiles_file_sep'])

        ## Get clipped
        if self.options['clipped_smiles_file'] == None:
            logging.info('clipping smiles file')
            bbSelectBuilder.enumerate_clipped_smiles(self.options['rxn_smarts'])
        else:
            logging.info(f"loading clipped smiles from {self.options['clipped_smiles_file']}")
            bbSelectBuilder.load_clipped_smiles_file(self.options['clipped_smiles_file'], sep = '\t')

        ## Get conformers
        if self.options['conformer_sdf'] == None and self.options['aligned_mols_sdf'] == None:
            logging.info('generating conformers')
            bbSelectBuilder.generate_conformers(
                output_sdf = self.options['conformer_sdf_save_location'],
                program = self.options['conformer_generation_program'],
                omega_load = self.options['omega_load_command'],
                omega_command = self.options['omega_run_command'],
                )
        else:
            logging.info(f"loading conformers from {self.options['conformer_sdf']}")
            bbSelectBuilder.load_conformer_sdf(self.options['conformer_sdf'])

        ## Align Mols
        if self.options['aligned_mols_sdf'] == None:
            bbSelectBuilder.align_mols(output_file_path = self.options['save_aligned_mols_sdf'])
        else:
            bbSelectBuilder.load_aligned_conformer_sdf(self.options['aligned_mols_sdf'])

        ## Get pharmacophore features
        logging.info('getting pharmacophore_features')
        bbSelectBuilder.get_pharmacophore_features()

        ## Generate fingerprints
        logging.info('generating fingerprints')
        bbSelectBuilder.get_fingerprint_dictionary()

        ## Write database
        logging.info('writing bbSelect DB')
        bbSelectBuilder.write_bbSelect_DB(self.options['output_root'])

        return bbSelectBuilder

class bbSelectDBbuilder():
    """
    This class wraps the bbGAP functions for straightforward usage
    """
    def __init__(self, cell_size, num_cells, pharmacophore_family_dict = 'default', ncpu = 0):
        
        # Set bbGAP parameters
        self._cell_size = cell_size
        self._num_cells = num_cells

        self._num_bits = num_cells * num_cells * 2
        
        self._mols = None
        self._IDs = None
        ## Deal with merging pharmacophores from rdkit
        if pharmacophore_family_dict == 'default':
            self._pharmacophore_family_dict = get_pharmacophore_family_dict()
        elif type(pharmacophore_family_dict) == dict:
                 self._pharmacophore_family_dict = pharmacophore_family_dict()
        else:
            raise ValueError(f'pharmacophore_family_dict should be either "default" or a dictionary')
        self._fp_size = len(set(self._pharmacophore_family_dict.values())) * self._num_bits
        self._excess_bits = 8 - (self._fp_size % 8)
        
        self._clipped_bb_smi_location = None
        self._smiles_file_location = None
        self._clipped_smiles_file_location = None
        self._conformer_sdf_location = None
        self._aligned_conformer_sdf_location = None
        self._mols_aligned = False
        
        ## Deal with parallelisation
        if ncpu > len(os.sched_getaffinity(0)) -1:
            logging.info(f'{ncpu} cores exceeds the number available. Setting to default')
            self._ncpu = 0
        if ncpu == 0:
            self._ncpu = max(1, len(os.sched_getaffinity(0)) -1 - 1)
        else:
            self._ncpu = ncpu
        
        self._unique_pharmacophores = list(dict.fromkeys(list(self._pharmacophore_family_dict.values())))            

        # This feature is to be deprecated
        self._use_rdkit = True

    def load_smiles_file(self, smiles_file_location, sep = ',', smiles_column_name = 'smiles'):
        """
        Define the location of the building block file.
        This must contain two columns: smiles and ID, in that order.
        Any additional data will also be imported.
        Args:
            smiles_file_location (str): location of the smiles file
            sep (str): delimiter for input file
            smiles_column_name (str): name of smiles column in file.
        """
        self._smiles_file_location = smiles_file_location
        self._smiles_file_sep = sep
        self._smiles_table =  pd.read_csv(self._smiles_file_location, sep = sep)
        self._smiles_table.rename(columns = {smiles_column_name: 'smiles'})
                
    def load_clipped_smiles_file(self, clipped_smiles_file, sep, clipped_smiles_column_name = 'clipped_smiles'):
        """
        Define the location of the clipped builing block smiles file
        """
        self._clipped_smiles_file_location = clipped_smiles_file
        self._clipped_smiles_table = pd.read_csv(self._clipped_smiles_file_location, sep = sep)
        self._clipped_smiles_table.rename(columns = {clipped_smiles_column_name: 'clipped_smiles'})
        
    def load_conformer_sdf(self, conformer_sdf_location):
        """
        Load conformers from an sdf file. 
        See load_sdf() function for more details
        """
        logging.info('loading conformer sdf')
        self._conformer_sdf_location = conformer_sdf_location
        self._mols, self._IDs = load_sdf(self._conformer_sdf_location)
        
    def load_aligned_conformer_sdf(self, aligned_conformer_sdf_location):
        """
        Load aligned conformers from an sdf file. Use if conformers have been aligned and saved 
        See load_sdf() function for more details
        """
        self._aligned_conformer_sdf_location = aligned_conformer_sdf_location
        self._mols, self._IDs = load_sdf(self._aligned_conformer_sdf_location)
        
        self._mols_aligned = True
        
    def enumerate_clipped_smiles(self, rxn_smarts, output_file = None):
        """
        Enumerate the input smiles file, given the reaction smarts, to give a clipped smiles file
        See function enumerate_smi_file() for more details
        """
        input_file = self._smiles_file_location
        
        if input_file == None:
            raise ValueError(f'Please load a smiles file using the bbGAP.load_smiles_file() function')
            
        if output_file == None:
            # If no output file is given, put it into the same place as the input file and put "_clipped" before the extension.
            input_file_portions = input_file.split('.')
            output_file = '.'.join(input_file_portions[:-1]) + '_clipped.tsv'
        else:
            pass
        
        # Run the enumeration
        enumerate_smi_file(input_file, output_file, rxn_smarts, sep = self._smiles_file_sep)
        logging.info(f"clipped smiles saved at {output_file}")
        self._clipped_smi_sep = '\t'
        
        self.load_clipped_smiles_file(output_file, sep = self._clipped_smi_sep)
    
    def generate_conformers(self, output_sdf = None, program = 'omega', omega_load = 'module load openeye', omega_command = 'omega2'):
        """
        Generate conformers for the clipped smiles.
        See the conformer_generation() function for more details about the arguments
        """
        
        if self._clipped_smiles_file_location == None:
            raise ValueError('Please load the clipped smiles file using the bbGAP.load_clipped_smiles_file() or the bbGAP.enumerate_clipped_smiles() function')
        
        # Replace the extension with _conformers.sdf for the output file
        if output_sdf == None:
            input_file_portions = self._clipped_smiles_file_location.split('.')
            output_sdf = '.'.join(input_file_portions[:-1]) + '_conformers.sdf'
                
        # Generate conformers
        conformer_generation(input_tsv = self._clipped_smiles_file_location, 
                             output_sdf = output_sdf, 
                             program = program, 
                             ncpu = self._ncpu, 
                             omega_load = omega_load, 
                             omega_command = omega_command)
        
        self._conformer_sdf_location = output_sdf
        
        self.load_conformer_sdf(self._conformer_sdf_location)
                
    def align_mols(self, output_file_path = None):
        """
        Aligns molecules 
        See align_mols_to_origin() function for more information
        Args:
            output_file_path (str): location where to store the sdf of aligned molecules. If None, no saving
        """
        
        ncpu = self._ncpu
        
        if self._mols == None or self._IDs == None:
            logging.error(f'No conformers have been loaded. Use the bbGAP.load_conformer_sdf() function to load')
            raise ValueError()
        if ncpu == 1:
            aligned_mols = align_mols_to_origin(self._mols, self._IDs)
        elif ncpu > 1:
            aligned_mols = align_mols_to_origin_parallel(self._mols, self._IDs, num_cores = self._ncpu)
        else:
            logging.error(f'ncpu {self._ncpu} is not a correct integer')
            raise VaueError()
            
        self._mols = [i[0] for i in aligned_mols]
        self._IDs = [i[1] for i in aligned_mols]

        
        # Write the files to SDF if required
        if output_file_path != None:
            logging.info(f'writing aligned mols to {output_file_path}')
            start = time.time()
            sdf_file_stream = Chem.SDWriter(output_file_path)
            for mol, ID in zip(self._mols, self._IDs):
                mol.SetProp('_Name', ID)
                sdf_file_stream.write(mol)
            sdf_file_stream.close()
            self._aligned_conformer_sdf_location = output_file_path
            end = time.time()
            logging.info(f'wrote aligned mols in {round(end - start, 2)} seconds')
        
        # Mark the molecules as having been aligned
        self._mols_aligned = True
               
    def get_pharmacophore_features(self, fdefName = None):
        """
        Get the pharmacophore feature placements.
        See the get_mols_pharmacophore_features() function for more information.
        Args:
            fdefName: The path to the fdef file. Leave as None for default
        """


        
        ncpu = self._ncpu
        
        if ncpu == 1:
            self._pharmacophore_features = get_mols_pharmacophore_features(
                mols = self._mols, 
                IDs = self._IDs, 
                cell_size = self._cell_size, 
                num_cells = self._num_cells,
                fdefName = fdefName)
        else:
            self._pharmacophore_features = get_mols_pharmacophore_features_parallel(
                mols = self._mols, 
                IDs = self._IDs, 
                cell_size = self._cell_size, 
                num_cells = self._num_cells, 
                fdefName = fdefName,
                num_cores = self._ncpu)
                  
    def get_fingerprint_dictionary(self, store_individual_pharmacophores = False):
        """
        Calculate the fingerprints given the pharmacophore feature placements.
        See the generate_bbgap_fingerprints() function for more information.
        """
        
        self._fingerprint_dictionary = generate_bbgap_fingerprints(
                                                     pharmacophore_features = self._pharmacophore_features,
                                                     pharmacophore_family_dict = self._pharmacophore_family_dict, 
                                                     num_cells = self._num_cells, 
                                                     store_individual_pharmacophores = store_individual_pharmacophores) # This is usually set to False, but True here for information purposes
    
    def write_bbSelect_DB(self, output_root):
        """
        Writes the bbGAP database, which consists of a binary file (.bin) and a reference file (.ref)
        These are used to store the results of bbgap.
        See the write_fingerprints_to_file() function to learn more
        Args:
            output_root (str): the location where the database will go. the DB files will use this root for .ref and .bin files
        """
        
        # Check that the directory can be written to
        path = "/".join(output_root.split('/')[:-1]) + "/"
        if not os.access(path, os.W_OK):
            raise PermissionError(f"cannot write to path {path}")
            
        self._bin_location = output_root + ".bin"
        self._ref_location = output_root + ".ref"
        
        # Write the fingerprints and capture their positions and the overall occupancy
        self._compound_bit_position_dict_list, self._overall_occupancy = write_fingerprints_to_file(self._bin_location, self._fingerprint_dictionary, self._fp_size, self._excess_bits)
        logging.info(f'Writing reference file')
        
        ## Create a DataFrame will all the required information
        
        # First add the IDs and the bit positions
        self._compound_data_table = pd.DataFrame.from_records(self._compound_bit_position_dict_list)
        
        # Then add the smiles and the clipped smiles.
        self._compound_data_table = self._compound_data_table.merge(self._smiles_table, how = 'left', on = 'ID').merge(self._clipped_smiles_table, how = 'left', on = 'ID')
        
        # Then add the conformer count from the list of IDs
        
        self._confs_count_table = count_confs_from_IDs(self._IDs)
        self._compound_data_table = self._compound_data_table.merge(self._confs_count_table, how = 'left', on = 'ID')
        self._compound_data_table = add_unique_pharmacophores_to_df(self._pharmacophore_features, self._compound_data_table)
        
        # To work with rest of bbGAP code, ID must be called "molname"
        self._ref_table = self._compound_data_table.rename(columns = {'ID': 'molname'}, inplace = False, copy = False)
        
        # Write the parameters of bbGAP to top of file
        with open(self._ref_location, "w") as ref_output:
            ref_output.write(str(self._overall_occupancy))
            ref_output.write(f'\npmap_area\t{self._num_bits}\n')
            ref_output.write(f'binary_file\t{self._bin_location}\n')
            ref_output.write(f'fp_size\t{self._fp_size+self._excess_bits}\n')
            ref_output.write(f'excess_bits\t{self._excess_bits}\n')
            ref_output.write(f'numX\t{self._num_cells*2}\n')
            ref_output.write(f'numY\t{self._num_cells}\n')
            for i, pharmacophore in enumerate(self._unique_pharmacophores):
                ref_output.write(f'pharmacophore\t{pharmacophore}\t{i}\n')
        # Write the data
        self._ref_table.to_csv(self._ref_location, sep = '\t', mode = 'a', index = False)
        
        logging.info('Reference file written')

    def GetBinLocation(self):
        return  self._bin_location
        
    def GetRefLocation(self):
        return self._ref_location

def conformer_generation(input_tsv, output_sdf, program, ncpu, omega_load = 'module load openeye', omega_command = 'omega2', **kwargs):
    """
    Run conformer generation using either omega or rdkit.
    Args:
        input_tsv (str): Path to tsv file containing smiles and ID
        output_sdf (str): Path to output sdf file
        program (str): Between 'omega' or 'rdkit'. Must have omega installed
        ncpu (int): number of cpu to use.
        omega_load (str): Command used to load omega in shell. If already loaded, use None
        omega_command (str): Command used to run omega in shell
        
    """
    if program == 'omega':
        start = time.time()
        # Load openeye tools and run omega2
        if omega_load != None:
            cmd = [f'{omega_load};']
        else:
            cmd = []
        cmd.extend([omega_command, '-flipper', 'true', '-in', input_tsv, '-out', output_sdf, '-mpi_np', f'{ncpu}'])
        cmd_string = " ".join(cmd)
        logging.info(f'Attempting to run the following command in shell:\n{cmd_string}')
        process  = subprocess.Popen(cmd_string, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        process.wait()
        if process.returncode:
            logging.info(f'Shell returned error code {process.returncode}. See below for stdout / stdin')
            logging.info(process.stderr.read().decode("utf-8"))
        else:
            end = time.time()
            logging.info(f'conformer generation with omega completed in {round(end - start, 2)} seconds')

    elif program == 'rdkit':

        from rdkit_conformer_ensemble_generator import conformer_ensemble_generator_parallel

        logging.info(f'Starting conformer generation with rdkit over {ncpu} cores.')

        input_table = pd.read_csv(input_tsv, sep = '\t')

        for column in ['smiles','ID']:
            if column not in input_table.columns:
                raise ValueError('input tsv for rdkit conformer generation must have {column} in column named "{column}"')

        conformer_ensemble_generator_parallel(
            smiles = input_table['smiles'],
            IDs = input_table['ID'],
            num_cores = ncpu,
            output_file = output.sdf,
            num_conf = options.num_conf,
            **kwargs)

        """
        Additional arguments:
        pruneRmsThresh = options.pruneRmsThresh,
        additional_optimisation = options.additional_optimisation,
        energy_threshold = options.energy_threshold,
        max_attempts = options.max_attempts
        """
    else:

        logging.error(f'{program} unsupported')

def add_unique_pharmacophores_to_df(pharmacophore_features, df):
    """
    Takes the calculated pharmacophore feature placement dictionary and identifies the unique pharmacophores
    This is then added to a provided dataframe, matched on ID
    Args:
        pharmacophore_features (list of dict): dictionary from get_mols_pharmacophore_features() function
        df (pd.DataFrame): DataFrame containing IDs. A column containing a dictionary printed as a string will be added.
    Returns:
        dict containing unique pharmacophores per ID
    """
    feature_count = {}
    for feature_dict in pharmacophore_features:
        if feature_dict == None:
            continue
        ID = feature_dict['ID']
        if ID not in feature_count.keys():
            feature_count[ID] = []
        feature_count[ID].append(feature_dict['Feature'])

    for ID in feature_count.keys():
        feature_count[ID] = set(feature_count[ID])

    ## Add features to the output table

    for i, row in df.iterrows():
        ID = row['ID']
    
        df.loc[i, 'pharmacophores'] = str(feature_count[ID])
        
    return df 


        
def count_confs_from_IDs(IDs):
    """
    Counts the number of conformers from the duplicates in a list of IDs after reading the SDF.
    In reality, because the stereoisomers of compounds with racemic centres are enumerated before conformer generation,
    this count is across all possible stereoisomers.
    Args:
        IDs (list): List of IDs from the sdf
    Returns:
        pd.DataFrame containing the ID and the number of conformers (confs)
    """
    conf_count_dict = {}
    for ID in IDs:
        if ID not in conf_count_dict.keys():
            conf_count_dict[ID] = 1
        else:
            conf_count_dict[ID] += 1
    conf_count_table = pd.DataFrame.from_dict(conf_count_dict, orient = 'index').reset_index()
    conf_count_table.columns = ['ID','confs']
    return conf_count_table

            
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

def get_simple_SMARTS_features():
    """
    Return a list of simple SMARTS definitions for pharmacophores
    """
    features = { 
        'Hydrophobic':"[$([C;H2,H1](!=*)[C;H2,H1][C;H2,H1][$([C;H1,H2,H3]);!$(C=*)]),$(C([C;H2,H3])([C;H2,H3])[C;H2,H3])]",
        'Donor': '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]',
        'Acceptor': '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]',
        'AromaticAttachment' : '[$([a;D3](@*)(@*)*)]',
        #'AliphaticAttachment': '[$([A;D3](@*)(@*)*)]',
        #'UnusualAtom': '[!#1;!#6;!#7;!#8;!#9;!#16;!#17;!#35;!#53]',
        'BasicGroup': '[$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([N,n;X2;+0])]',
        'AcidicGroup': '[$([C,S](=[O,S,P])-[O;H1])]',
    }
    return features

def align_mol_to_origin(mol, alignment_smarts_mol):
    """
    Function which aligns the conformers of an inputted rdkit mol object where the first atom in the alignment smarts
    sits at 0,0,0 and the second atom sits on the x-axis: y=0, z=0.
    
    Args:       
        mol (rdkit:Mol object): molecule to align. Must contain one or more conformers
        alignment_smarts_mol (rdkit:Mol object): SMARTS string in Mol object to anchor the alignment to.
        
    Returns:
        list of mol objects which contain a single conformer each, where the conformers are aligned as detailed above
    """
    
    # Create list that will hold the aligned conformers
    aligned_confs = []
    
    # Check the mol object contains conformers
    
    # Number of conformers
    #mol.GetConformers
    if len(mol.GetConformers()) == 0:
        raise ValueError("molecule does not contain conformers.")
    
    # Define substructure
    substructure = alignment_smarts_mol
    
    # Find substructure in molecule
    matches = mol.GetSubstructMatches(substructure)
    
    if not matches:
        raise ValueError("Substructure of attachment point not found")
        
    # For each conformer in the molecule
    
    aligned_mols = []
    for conf_id in range(len(mol.GetConformers())):
        
        # Take a copy of the molecule, containing only the conformer in question
        mol_copy = Chem.Mol(mol, False, conf_id)
        
        # Get the conformer
        conf = mol_copy.GetConformer()

        # Get the atom indices of [15C] and its adjacent atom
        idx_15C, idx_adjacent = matches[0]

        ## Translate molecule so 15C is at origin

        # Get the position of the 15C atom
        pos_15C = conf.GetAtomPosition(idx_15C)

        # Its translation is simply a movement to 0,0,0
        translation = (-pos_15C.x, -pos_15C.y, -pos_15C.z)

        # Move all atoms in the molecule according to the translation
        for atom_id in range(mol_copy.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_id)
            conf.SetAtomPosition(atom_id, (pos.x + translation[0], pos.y + translation[1], pos.z + translation[2]))

        ## Rotate the adjacent atom of 15C to lie on the x-axis

        # Get the position of the atom next to the 15C
        neighbor_pos = np.array(conf.GetAtomPosition(idx_adjacent))

        # Get the direction in which that atom would need to travel to get to the x-axis
        direction = neighbor_pos / np.linalg.norm(neighbor_pos)  # normalized direction

        # Calculate the rotation axis and angle
        rotation_axis = np.cross(direction, [1, 0, 0])
        rotation_angle = np.arccos(np.dot(direction, [1, 0, 0]))
        rot_mat = rotation_matrix(rotation_axis, rotation_angle)

        # Apply the rotation to all atoms in the molecule
        for atom_id in range(mol_copy.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(atom_id))
            new_pos = np.dot(rot_mat, pos)
            conf.SetAtomPosition(atom_id, new_pos)

        # Check if the adjacent atom's x-coordinate is negative after rotation.
        # If so, apply a 180-degree rotation around the z-axis to flip the molecule.
        # This is so all the adjacent atoms are overlaid
        if conf.GetAtomPosition(idx_adjacent).x < 0:
            rot_mat_180 = rotation_matrix([0, 0, 1], np.pi)
            for atom_id in range(mol_copy.GetNumAtoms()):
                pos = np.array(conf.GetAtomPosition(atom_id))
                new_pos = np.dot(rot_mat_180, pos)
                conf.SetAtomPosition(atom_id, new_pos)
                
        aligned_mols.append(mol_copy)
        
    return aligned_mols

def align_mols_to_origin(mols, IDs, alignment_smarts = '[15C]*'):
    """
    Function which wraps the align_molecule_confs_to_origin function to work efficiently for a list of mols.
    
    Args:
        mols (list of rdkit:Mol objects): list of molecules to align. Each molecule must contain one or more conformers
        alignment_smarts (str): SMARTS string detailing the attachment atom and the adjacent atom
        
    Returns:
        list of mol objects which contain a single conformer each, where the conformers are aligned
    """
    
    # Import SMARTS as Mol Object
    substructure = Chem.MolFromSmarts(alignment_smarts)
    
    aligned_mol_list = []
    ID_list = []
    
    # Iterate over mols and align their conformers.
    for mol, ID in zip(mols, IDs):
        aligned_mol = align_mol_to_origin(mol, substructure)
        aligned_mol_list.extend([[mol, ID] for mol in aligned_mol])
    
    return aligned_mol_list

def chunk_list(lst, n):
    """Split the list into n chunks.
    Args:
        lst (list): list to split into chunks
        n (integer): number of chunks to make
    Returns:
        list containing chunks of original list"""
    
    return [lst[i::n] for i in range(n)]

def align_mols_to_origin_parallel(mols, IDs, alignment_smarts = '[15C]*', num_cores = 0):
    """
    Aligns molecules (conformers) over multiple cores using the multiprocessing package in python
    Uses the align_mols_to_origin function, which uses the align_mol_to_origin function
    
    Args:
        mols (list of rdkit.Mol): list of compounds to align
        alignment_smarts (str): SMARTS of where to align the molecules
        
    Returns:
        list of aligned molecules        
    """
    start = time.time()
    
    # Check that the requested number of cores is avaleble, if not just set to default (n-1)
    if num_cores != 0:
        if num_cores > len(os.sched_getaffinity(0)) -1:
            logging.info(f'{num_cores} cores exceeds the number available. Setting to default')
            num_cores = 0
        else:
            pass
    if num_cores == 0:
        num_cores = max(1, len(os.sched_getaffinity(0)) - 1)  # Use all cores except one
        
    logging.info(f'Running alignment over {num_cores} cores')
    
    # Chunk the list of molecules
    mols_chunks = chunk_list(mols, num_cores)
    IDs_chunks = chunk_list(IDs, num_cores)
    # Run alignment over multiple cores
    with multiprocessing.Pool(num_cores) as pool:
        # Process each chunk of molecules
        #aligned_mols_chunks = pool.starmap(align_mols_to_origin, [(chunk, alignment_smarts) for chunk in mols_chunks])
        aligned_mols_chunks = pool.starmap(align_mols_to_origin, zip(mols_chunks, IDs_chunks, [alignment_smarts]*num_cores))

    # Flatten the list of aligned molecules
    aligned_mols = [mol for chunk in aligned_mols_chunks for mol in chunk]

    end = time.time()
    logging.info(f'{len(mols)} compounds aligned over {num_cores} CPU cores in {end - start} seconds')     
    return aligned_mols

def get_pharmacophore_dict(feature_dict = None):
    """
    Creates the pharmacophore dictionary from a feature dictionary, containing rdkit mol objects for the smarts
    
    Args:
        feature_dict (dict): dictionary keyed by pharmacophore name and containing pharmacophore SMARTS definition
    
    Returns
        dict: dictionary keyed by pharmacophore name containing 'smarts' with a SMARTS string and 'pattern' with the rdkit mol object for the smarts
    """
    pharmacophore_dict = {}
    
    # Use default if no feature dictionary is given
    if feature_dict == None:
        features = get_simple_SMARTS_features()
    else:
        features = feature_dict
    
    # Iterate through feature dict, creating the rdkit:Mol object
    for feature in features.keys():
        pharmacophore_dict[feature] = {}
        pharmacophore_dict[feature]['smarts'] = features[feature]
        pharmacophore_dict[feature]['pattern'] = Chem.MolFromSmarts(features[feature])
        
    return pharmacophore_dict
        
    
def get_mol_pharmacophore_features(mol, ID, num_cells, cell_size, pharmacophore_family_dict, pharmacophore_dict = None, feature_factory = None, use_rdkit = True, store_all = False):
    """
    Find the pharmacophore features in the molecule and return the features, their 3D coordinates, their distance from the origin, the angle from the x axis and the resulting 2D coordinates
    Args:
        ID (str): ID of molecule. If None is entered, the SMILES will be used
        mol (rdkit.Mol object): molecule object containing a single conformer
        num_cells (int): number of cells in the positive x-direction to use
        cell_size (float): size of cell in square angstroms
        pharmacophore_family_dict (dict): dictionary detailing how to combine pharmacophore features, if needed. Otherwise False if not used
        pharmacophore_dict: dictionary keyed by pharmacophore name containing 'smarts' with a SMARTS string and 'pattern' with the rdkit mol object for the smarts
        feature_factory: RDkit Chem.ChemicalFeatures.Factory object. If not using, select None
        use_rdkit (bool): Whether to use rdkit feature factory to capture pharmacophores

    Returns:
        list of dictionaries containing the ID, Pharmacophore, Atom, 3D coordinates, distance from origin, angle from x-axis and resulting 2D coordinates
    """
    
    feature_list = []
    
    if ID == None:
        ## If no ID is given, assign the SMILES as the ID
        ID = Chem.MolToSmiles(mol)
    
    ## This is a simple method which uses a SMARTS match to find the locations of the pharmacophores
    if not use_rdkit:
        feature_atoms = {}
        # Identify features in the molecule
        # Iterate through pharmacophore defs and find matches in the molecules
        
        for pharmacophore in pharmacophore_dict:
            pharmacophore_matches = mol.GetSubstructMatches(pharmacophore_dict[pharmacophore]['pattern'])
            if pharmacophore_matches:
                feature_atoms[pharmacophore] = [idx for match in pharmacophore_matches for idx in match]

        # Tabulate coordinates for each feature in each conformer
        conf = mol.GetConformer()
        for feature_name, atom_indices in feature_atoms.items():
            #print(f"  {feature_name}:")
            for idx in atom_indices:
                # Get the atom position
                pos = conf.GetAtomPosition(idx)
                
                ## Transform the position into a standard dictionary for capturing pharmacophore placement
                feature_dict = get_feature_placement_dict(pos = pos, feature_name = feature_name, ID = ID, num_cells = num_cells, cell_size = cell_size, store_all = store_all)
                if feature_dict == None:
                    continue
                else:
                    feature_list.append(feature_dict)
    
    ## This method uses the built-in rdkit pharmacophore feature capture method
    ## This is slightly better because it accounts for the aromatic centroid, for example, and captures far more pharmacophore examples
    else:
        # Use rdkit featureFactory to find the features within a molecule
        feats = feature_factory.GetFeaturesForMol(mol)
        unique_pharmacophore_features = list(dict.fromkeys(list(pharmacophore_family_dict.values())))

        for f in feats:
            if pharmacophore_family_dict:
                if f.GetFamily() not in unique_pharmacophore_features:
                    continue
            #print(f.GetFamily(), f.GetType(), f.GetPos().x, f.GetPos().y, f.GetPos().z)
            # Transform the features into a standard dictionary for capturing pharmacophore placement
            feature_dict = get_feature_placement_dict(pos = f.GetPos(), feature_name = f.GetFamily(), ID = ID, num_cells = num_cells, cell_size = cell_size, store_all = store_all)
            feature_list.append(feature_dict)
          
    return feature_list
            
def get_feature_placement_dict(pos, feature_name, ID, num_cells, cell_size, store_all = False):
    """
    Function to calculate the projected 2D placement of a pharmacophore feature and return in a standardised dictionary.
    
    Args:
        pos (array): an object which contains the x,y,z coordinates of a pharmacophore placements as self.x, self.y, self.z respectively 
        feature_name (str): the name of a pharmacophore
        ID (str): the ID of a molecule
        num_cells (int): number of cells in the positive x-direction to use
        cell_size (float): size of cell in square angstroms
    
    Returns:
        return a dictionary containing the ID, feature, name, 3D position, distance from the origin, angle from the x-axis and projected 2D coordinates
    """
    
    # calculate the distance from the origin
    distance_from_origin = np.linalg.norm(pos)

    # Calculate the angle from the x-axis

    angle_from_x = np.degrees(np.arccos(pos[0] / distance_from_origin))

    # Calculate the x and y on a 2D grid using the angle and distance

    new_x_coordinate = distance_from_origin * math.cos(math.radians(angle_from_x))
    new_y_coordinate = distance_from_origin * math.sin(math.radians(angle_from_x))
    
    cell_info = get_cell_numbers(x_2d = new_x_coordinate, y_2d = new_y_coordinate, cell_size = cell_size, num_cells = num_cells, ID = ID)
    
    # Check that the calculation ran correctly
    if cell_info == None:
        return None
    
    else:
        x_cell, y_cell, cell_number = cell_info

    feature_dict = {'ID':ID,
                    'Feature': feature_name,
                    'cell_number' : cell_number,
                    }
    
    # To save storage space, the additional geometric parameters are not stored.
    # This is turned back on by creating a second dictionary and using it to update the returned dictionary
    if store_all:
        additional_dict = {
                        'x_3d': pos.x,
                        'y_3d': pos.y,
                        'z_3d': pos.z,
                        'distance_from_origin': distance_from_origin,
                        'angle_from_x_axis': angle_from_x,
                        'x_2d' : new_x_coordinate,
                        'y_2d' : new_y_coordinate,
                        'x_cell' : x_cell,
                        'y_cell' : y_cell,}
        feature_dict.update(additional_dict)
    
    
    #print(f"    Atom {idx}: x = {pos.x:.2f}, y = {pos.y:.2f}, z = {pos.z:.2f}, distance = {distance_from_origin}, angle = {angle_from_x}, x = {new_x_coordinate}, y = {new_y_coordinate}")
    
    return feature_dict

def get_cell_numbers(x_2d, y_2d, cell_size, num_cells, ID):
    """
    Converts coordinates on x,y to a cell number. This starts at 1, increases along the x axis, followed by the y axis
    For example:
    
        7  8  9
        4  5  6
        1  2  3
    
    Args:
        x_2d: float
            coordinates on the x-axis
        y_2d: float
            coordinates on the y-axis
        cell_size: float
            size of the cell (in Angstroms)
        num_cells: int
            number of cells in the positive direction in the x axis
            
    Returns: list 
        containing the cell in the x-axis and y-axis, and the cell number
    
    """
    
    limit = num_cells * cell_size

    if x_2d > limit * 2 or y_2d > limit:
        
        logging.warning(f"Pharmacophore feature point ({x_2d, y_2d}) Angstroms for cmpd {ID} is greater than pharmacophore capture area. Set a higher number of cells if you want to capture this pharmacophore feature. Skipping...")
        return None
    
    # transform x and y relative to the cell size
    x_trans = x_2d / cell_size
    y_trans = y_2d / cell_size
    
    # Shift the values based on the number of cells (this puts the origin in the centre of the x-axis)
    x_shift = x_trans + num_cells
    y_shift = y_trans
    
    # Get the cell coordinate
    x_cell = np.floor(x_shift) + 1
    y_cell = np.floor(y_shift) + 1
    
    # Calculate the cell number
    cell_number = x_cell + ( (y_cell - 1) * ( 2 * num_cells ))
    
    # Check if the cell number is greater than permitted
    if cell_number > 2 * num_cells**2:
        logging.error(f"Calculation wrong for id {ID}")
        
    return x_cell, y_cell, cell_number
    
def get_mols_pharmacophore_features(mols, IDs, cell_size, num_cells, fdefName, use_pharmacophore_family_dict = True, pharmacophore_features = None, use_rdkit = True, store_all = False):
    """
    Wraps the get_mol_pharmacophore_features() and the pharmacophore_dict() functions to return pharmacophore features for a list of molecules.
    
    Args:
        mols (list of RDkit:mol objects): molecules to process. These have to have been aligned already
        IDs (list of str): IDs to process, in exact order as molecules. Supply empty list if none are available
        num_cells (int): number of cells in the positive x-direction to use
        cell_size (float): size of cell in square angstroms        
        fdefName (str or None): Path to pharmcophore feature definitions. Use None for default
        use_pharmacophore_family_dict (Bool): Whether to use the pharmacophore family dictionary to simply the pharmacophores
        pharmacophore_features (dict, str, or None): pharmacophore feature dictionary, 'rdkit', or None to use simple pharmacophores
        use_rdkit (Bool): Whether to use rdkit feature factory to define pharmacophores
        store_all (Bool): Whether to store all geometric parameters

        
    Returns:
        list of dictionaries containing feature dictionaries
    """
    if fdefName == None:
        fdefName = str(files('data').joinpath('BaseFeatures_bbGAP.fdef'))


    # If using self-defined SMARTS, set the parameters
    if not use_rdkit:
        pharmacophore_dict = get_pharmacophore_dict(pharmacophore_features)
        feature_factory = None
        
    # Otherwise, initiate the feature factory using rdkits default feature definitions
    # If using other definitions, this part of the code would need to be updated.
    else:
        pharmacophore_dict = None
        # Set up a FeatureFactory once that can be used for the whole set
        # Using edited fdef file
        #fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        #fdefName = '../data/BaseFeatures_bbGAP.fdef'
        feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    
    if use_pharmacophore_family_dict:
        pharmacophore_family_dict = get_pharmacophore_family_dict()
    else:
        pharmacophore_family_dict = False
        
    pharmacophore_features = []
    
    # If no IDs are given supply None to the function, create a list containing "None"
    if len(IDs) == 0:
        IDs = [None] * len(mols)
        
    # If IDs are given, check whether there are the correct number of them and supply to the function
    else: 
        if len(mols) != len(IDs):
            raise ValueError(f"Number of mols ({len(mols)}) and IDs ({len(IDs)}) is not the same")
        
    for mol, ID in zip(mols, IDs):
        
        features = get_mol_pharmacophore_features(
            mol = mol, 
            ID = ID, 
            cell_size = cell_size, 
            num_cells = num_cells,
            pharmacophore_family_dict = pharmacophore_family_dict,
            pharmacophore_dict = pharmacophore_dict, 
            feature_factory = feature_factory, 
            use_rdkit = use_rdkit,
            store_all = store_all)
        
        pharmacophore_features.extend(features)
    
    return pharmacophore_features
                                    
def get_mols_pharmacophore_features_parallel(mols, IDs, cell_size, num_cells, fdefName = None, use_pharmacophore_family_dict = True, pharmacophore_features = None, use_rdkit = True, store_all = False, num_cores = 0):
    """
    Generates pharmacophores over multiple cores using the multiprocessing package in python
    Uses the get_mols_pharmacophore_features function, which uses the get_mol_pharmacophore_features function
    
    Args:
        mols (list of RDkit:mol objects): molecules to process. These have to have been aligned already
        IDs (list of str): IDs to process, in exact order as molecules. Supply empty list if none are available
        num_cells (int): number of cells in the positive x-direction to use
        cell_size (float): size of cell in square angstroms        
        use_pharmacophore_family_dict (Bool): Whether to use the pharmacophore family dictionary to simply the pharmacophores
        pharmacophore_features (dict, str, or None): pharmacophore feature dictionary, 'rdkit', or None to use simple pharmacophores
        use_rdkit (Bool): Whether to use rdkit feature factory to define pharmacophores
        store_all (Bool): Whether to store all geometric parameters
        num_cores (int): Number of cores to use. 0 will use all available less one
        
    Returns:
        list of aligned molecules        
    """
    start = time.time()
    
    # Check that the requested number of cores is avaleble, if not just set to default (n-1)
    if num_cores != 0:
        if num_cores > len(os.sched_getaffinity(0))-1:
            logging.info(f'{num_cores} cores exceeds the number available. Setting to default')
            num_cores = 0
        else:
            pass
    if num_cores == 0:
        num_cores = max(1, os.cpu_count() - 1)  # Use all cores except one
        
    logging.info(f'Capturing pharmacophores over {num_cores} cores')
    
    # Chunk the list of molecules
    mols_chunks = chunk_list(mols, num_cores)
    IDs_chunks = chunk_list(IDs, num_cores)
        
    with multiprocessing.Pool(num_cores) as pool:
        
        # Using starmap to pass multiple arguments to the function.
        # Need to make a list creating values for each of the parameters that will be run on each of the functions across the CPUs
        results = pool.starmap(
            get_mols_pharmacophore_features, 
            zip(mols_chunks, 
                IDs_chunks,
                [cell_size]*num_cores,
                [num_cells]*num_cores,
                [fdefName]*num_cores,
                [use_pharmacophore_family_dict]*num_cores,
                [pharmacophore_features]*num_cores, 
                [use_rdkit]*num_cores,
                [store_all]*num_cores,
               ))

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]
    
    end = time.time()
    logging.info(f'{len(mols)} pharmacophores captured over {num_cores} cores in {end - start} seconds')
    
    return flattened_results
    
def draw_pharmacophore_atoms_for_mol(mol):
    """
    Function draws a molecule out and highlights the atoms for which the pharmacophore features have been associated.
    
    Args:
        mol : rdkit.Mol
            Molecule for which features have been calculated.
            
        feats: tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features generated for the molecule
            
    Returns: SVG
        Picture of the molecule highlighted with its pharmacophore features
            
    """
    # Using edited fdef file
    #fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    fdefName = './data/BaseFeatures_bbGAP.fdef'
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    atoms_list = {}
    for i, feat in enumerate(feats):
        atom_ids = feat.GetAtomIds()
        #feat_type = feat.GetType()
        feat_type = feat.GetFamily()
        #atoms_list[feat_type] = atom_ids
        atoms_list[i] = {}
        atoms_list[i]['feat_type'] = feat_type
        atoms_list[i]['atom_ids'] = atom_ids
    return Chem.Draw.MolsToGridImage([mol]*len(atoms_list), legends=[atoms_list[i]['feat_type'] for i in atoms_list.keys()], highlightAtomLists=[atoms_list[i]['atom_ids'] for i in atoms_list.keys()])

def get_pharmacophore_family_dict():
    """
    Defines how to categorise the default rdkit pharmacophores into custom pharmacophores
    By default, this will combine the Hydrophobe and LumpedHydrophobe pharmacophores
    and remove the ZnBinder pharmacophore.
    
    Also defines the order of the pharmacophores within the fingerprint by the order of the keys
    
    Can be customised if needed to change to aliases etc.
    
    Returns: dict
        dict defining how to categorise the phramcophores
    """

    return {
        'Donor' : 'Donor',
        'Acceptor' : 'Acceptor',
        'NegIonizable' : 'NegIonizable',
        'PosIonizable' : 'PosIonizable',
        'Aromatic' : 'Aromatic',
        'Hydrophobe' : 'Hydrophobe',
        'LumpedHydrophobe' : 'Hydrophobe',
        }


def enumerate_mol(smiles, rxn):
    """
    Args:
        smiles (str) : SMILES for molecule to run enumeration on
        rxn_mol (rdkit.Chem.Mol) : reaction SMARTS
    Returns:
        list of rdkit.Chem.Mol objects which represent the products from the enumeration
    """
    return list(itertools.chain.from_iterable(rxn.RunReactants([Chem.MolFromSmiles(smiles)])))

def enumerate_smi_file(input_file, output_file, rxn_smarts, sep = '\t'):
    """
    Enumerates molecules contained within a smiles file to an output smiles file
    Writes a header line containing smiles and ID
    
    Args:
        input_file: location of input smiles file. must contain smiles in first column and ID in second column
        output_file: location of output smiles file. will contain smiles in first column and ID in second column
        rxn_smarts: the SMARTS string detailing the reaction
    Returns: None
    """
    
    start = time.time()
    
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)

    with open(input_file,'r') as input_smi:

        with open(output_file,'w') as output_smi:

            output_smi.write('clipped_smiles\tID\n')

            for i, line in enumerate(input_smi):

                if i ==5:
                    pass
                    #break

                line_items = line.split(sep)

                if "smiles" in line_items[0].lower():
                    continue
                else:
                    smiles = line_items[0]
                    ID = line_items[1]
                    product_list = enumerate_mol(smiles, rxn)

                    if len(product_list) == 0:
                        print(f'smiles {i} cannot be reacted.')
                        continue

                    else:
                        for mol in product_list:
                            output_smi.write(f'{Chem.MolToSmiles(mol)}\t{ID}\n')
                            
            line_count = i + 1


    end = time.time()
    logging.info(f'{i} compounds clipped in {end - start} seconds')

def write_mols_to_sdf(mols, output_location):
    """
    Writes a list of mol objects to an SDF (V3000)
    Args:
        mols (list of rdkit.Mol objects): the molecules to write
        output_location (str): The file to which to save the molecules
    """
    
    writer = Chem.SDWriter(output_location)
    writer.SetForceV3000(1)
    for mol in mols:
        writer.write(mol)
    write.close()
    
def generate_bbgap_fingerprints(pharmacophore_features, pharmacophore_family_dict, num_cells, store_individual_pharmacophores = False):

    """
    Generate the bbGAP fingerprints for IDs, with their corresponding pharmacophore features and the cell numbers where the pharmacophore features are placed.
    The lists must be in the correct order and of the same length. So it is better to just feed columns from the pd.DataFrame
    
    Args:
        pharmacophore_features (list of dict): list of dictionaries with ID, pharmacophore feature, and cell number as keys
        pharmacophore_family_dict (dict): Dictionary which details how the rdkit pharmacophore feature families should be collapsed. This is important for the size of the fingerprint.
        num_cells (int): the number of cells to capture in the positive x-axis.
        store_individual_pharmacophores (bool): Whether to store the individual fps for the pharmacophores (before they are concatenated to give the final fingerprint)
    
    Returns:
        dictionary keyed by ID containing the fingerprint.
        """
    start = time.time()
    # Calculate the number of bits per pharmacophore feature in the fingerprint
    num_bits = num_cells * num_cells * 2

    # Get the unique pharmacophores that are to be captured, preserving the order they appear in the dictionary
    unique_pharmacophores = list(dict.fromkeys(list(pharmacophore_family_dict.values())))

    # Start a dictionary, keyed by molecule 
    compound_fingerprint_dict = {}
    
    # Iterate over each of the features
    for i in pharmacophore_features:
    # Iterate over each row in the feature table and generating the fingerprints
        if i == None:
            continue
            
        ID = i['ID']
        cell_number = i['cell_number']
        feature = i['Feature']
        
        index = unique_pharmacophores.index(feature)

        # Create dictionary entries keyed by ID
        if ID not in compound_fingerprint_dict.keys():
            compound_fingerprint_dict[ID] = {}
            
            if store_individual_pharmacophores:
                # For each pharmacophore, create the empty fingerprints
                for pharmacophore_feature in unique_pharmacophores:
                    compound_fingerprint_dict[ID][pharmacophore_feature] = np.zeros(num_bits)
                    
            # Create the overall fingerprint
            compound_fingerprint_dict[ID]['fp'] = ['0'] * num_bits * len(unique_pharmacophores)

        if store_individual_pharmacophores:
            # Turn on the bits for each pharmacophore feature fingerprint portion
            compound_fingerprint_dict[ID][feature][int(cell_number-1)] = True
        
        # Also turn on the overall fingerprint
        try:
            compound_fingerprint_dict[ID]['fp'][int((index * num_bits) + cell_number)] = '1'
        except:
            print(ID, i)
        
    end = time.time()
    logging.info(f'{len(compound_fingerprint_dict.keys())} fingerprints generated over 1 CPU core in {end - start} seconds')
    
    return compound_fingerprint_dict
                             
def load_sdf(sdf_file_location):
    """
    Loads an SDF using rdkit but without loading into a pd.DataFrame.
    This is much faster and uses a magnitude less memory
    
    Args:
        sdf_file_location (str): where the SDF file is stored
    Returns:
        list of two lists [mols, IDs]
    """
    start = time.time()
    with open(sdf_file_location, 'r') as sdf:

        # Capture mols and IDs from the file
        mols = []
        IDs = []

        # Start a container for a single molblock
        mol_file_lines = []

        for line in sdf:

            # When reaching the end of a molblock, convert the molblock to a mol
            if line.strip() == '$$$$':
                mol = Chem.MolFromMolBlock(''.join(mol_file_lines))
                ID = mol_file_lines[0].strip()
                mols.append(mol)
                IDs.append(ID)
                # Empty the container for the molblock lines
                mol_file_lines = []
            else:
                # Capture the lines of the molblock
                mol_file_lines.append(line)

    end = time.time()
    logging.info(f'{len(mols)} conformers loaded in {end - start} seconds')
    return mols, IDs

def write_fingerprints_to_file(binary_file, fingerprint_dict, fp_size, excess_bits):
    
    start = time.time()
    compound_bit_position_dict_list = []
    overall_occupancy = [0] * fp_size

    with open(binary_file, "wb") as out_bin:

        for pos, ID in enumerate(fingerprint_dict.keys()):

            fp = fingerprint_dict[ID]['fp']

            for i, bit in enumerate(fp):

                if bit == '1':
                    overall_occupancy[i] += 1

            fp = ''.join(fingerprint_dict[ID]['fp'])
            fp = fp + "".join([str(0)]*excess_bits) # Add excess bits
            fp_binary = bitarray(fp)
            fp_binary.tofile(out_bin)
            bit_count = fp_binary.count(1)
            position = pos * (fp_size + excess_bits)

            compound_bit_position_dict_list.append({'ID' : ID, 'fp_start_position' : position, 'bit_count' : bit_count})


    end = time.time()
    logging.info(f'{len(fingerprint_dict.keys())} fingerprints written in {end-start} seconds')
    
    return compound_bit_position_dict_list, overall_occupancy

def main():


    # Parse arguments

    usage = "Generate bbSelect database."

    parser = argparse.ArgumentParser(description = usage,
                                     formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument("--smiles", 
        action = "store", 
        type = str,
        dest = "input_file",
        help = "Destination of input file containing columns 'smiles' and 'ID'",
        required = True)

    parser.add_argument("--out",
        action = "store",
        type = str,
        dest = "output_file",
        help = "ROOT name of the output .bin and .ref files",
        required = True)

    parser.add_argument("--smiles_column",
        action = "store",
        type = str,
        dest = "smiles_column",
        help = "Column name (or index) where smiles exist, Default: %(default)s",
        default = "smiles")

    parser.add_argument("--ID_column",
        action = "store",
        type = str,
        dest = "ID_column",
        help = "Column name (or index) where IDs exist, Default: %(default)s",
        default = "ID")

    parser.add_argument("--num_conf",
        action = "store",
        type = int,
        dest = "num_conf",
        help = "Max number of conformers to generate, Default: %(default)s",
        default = 400)

    parser.add_argument("--pruneRmsThresh",
        action = "store",
        type = int,
        dest = "pruneRmsThresh",
        help = "RMS threshold for pruning, Default: %(default)s",
        default = 0.5)

    parser.add_argument("--ncpu",
        action = "store",
        type = int,
        dest = "ncpu",
        help = "Number of CPU cores to use. 0 will use the max available - 1. Default: %(default)s",
        default = 0)

    parser.add_argument("--sep",
        action = "store",
        type = str,
        dest = "sep",
        help = "Dilimiting character for input file. Default: tab '\\t'",
        default = '\t')

    parser.add_argument("--opt",
        action = "store_true",
        dest = "additional_optimisation",
        help = "Whether to run additional optimisation. Default: False",
        default = False)

    parser.add_argument("--energy_threshold",
        action = "store",
        type = int,
        dest = "energy_threshold",
        help = "Energy threshold for conformers. Not recommended. Default: 0",
        default = 0)

    parser.add_argument("--max_attempts",
        action = "store",
        type = int,
        dest = "max_attempts",
        help = "Maximum attempts during distance geometry conformer generation",
        default = 1000)

    options = parser.parse_args()

    logging.basicConfig(format = '%(levelname)s -%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.INFO)

    if not os.access(options.input_file,os.R_OK) or not os.path.isfile(options.input_file):
        raise ValueError(f'input file:{options.input} not accessible')

    input_table = pd.read_csv(options.input_file, sep = options.sep)

    if options.ncpu <= 0:
        num_cores = len(os.sched_getaffinity(0)) -1
    elif options.ncpu > len(os.sched_getaffinity(0)) -1:
        logging.warning(f'{options.ncpu} cores not available. running with {len(os.sched_getaffinity(0)) -1}')
        num_cores = len(os.sched_getaffinity(0)) -1

    else:
        num_cores = options.ncpu

    start = time.time()
    logging.info(f'Starting conformer generation over {num_cores} cores.')

    conformer_ensemble_generator_parallel(
        smiles = input_table[options.smiles_column],
        IDs = input_table[options.ID_column],
        num_conf = options.num_conf,
        pruneRmsThresh = options.pruneRmsThresh,
        additional_optimisation = options.additional_optimisation,
        num_cores = num_cores,
        energy_threshold = options.energy_threshold,
        max_attempts = options.max_attempts,
        output_file = options.output_file)

    end = time.time()

    logging.info(f'{len(input_table)} compounds processed in {round((end-start) / 60, 2)} minutes')   

if __name__ == '__main__':
    main()

