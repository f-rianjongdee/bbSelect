#!/usr/bin/env python

"""
Contains the functions to process the bbGAP database


Adapted and expanded from the below resources:

https://github.com/dkoes/conformer_analysis/blob/main/gen_rdkit_conformers.py
https://pubs.acs.org/doi/10.1021/acs.jcim.3c01245    
https://gist.github.com/tdudgeon/b061dc67f9d879905b50118408c30aac
https://iwatobipen.wordpress.com/2021/01/31/generate-conformers-script-with-rdkit-rdkit-chemoinformatics/

Notes:

# 252 seconds for 1 core, 100 compounds, gives only 21 passed. Energy threshold = 12.
# 132 seconds for 2 cores, 100 compounds, gives 21 passed. Energy threshold = 12.
# 27 seconds for 23 cores. 100 compounds, gives 100 passed. Energy threshold = 0.
# 27 seconds for 23 cores. 100 compounds, gives 37 passed. Energy threshold = 20.
    # Energy threshold seems to filter compounds rather than optimise the conformer ensembles. 
# 60 hours for 29 cores. 20,000 compounds with 1000 max attempts and 400 num_conf. No addn opt
# 14 hours for 19 cores. 20,000 compounds with 200 max attempts and 100 num_conf. No addn opt
# 14 hours for 19 cores. 20,000 compounds with 200 max attempts, 100 num_conf and addn opt

@author: Francesco Rianjongdee

Built on rdkit version 2022.03.1b1
"""


from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import multiprocessing
import os
import matplotlib.pyplot as plt
import logging
import time
import pandas as pd
import multiprocessing
import traceback
import logging
import argparse

def get_conformers_from_smiles(smiles, pruneRmsThresh, num_conf = 100, energy_threshold = 0, max_attempts = 200, additional_optimisation = False):

    # Get molecule
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    Chem.SanitizeMol(molecule)

    # Make initial conformer ensemble. This has an RMS pruning step.
    
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = pruneRmsThresh
    params.randomSeed = 0xf00d
    params.maxAttempts = max_attempts
    params.optimizerForceTol = 0.0135 # Taken from rdkit blog - found to give an increase in calculation speed
    
    # Get conformers for all stereoisomers if undefined.
    mols = EnumerateStereoisomers(molecule)
    
    return_mols = []
    return_errors = []
    
    for mol in mols:
        
        # Generate conformers
        try:
            cids = rdDistGeom.EmbedMultipleConfs(mol, num_conf, params)
    
        except:
            return_errors.append(Chem.MolToSmiles(mol))
            traceback.print_exc()
            continue
            
                
        if additional_optimisation == False:
            
            return_mols.append(mol)
        
        # Additional optimisation will optimise using MMFF and remove things within the specified RMS threshold, discarding higher energies first.
        
        else:
            
            # Perform optimisation on these and save the energies
            try:
                energies = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
            except:
                return_errors.append(Chem.MolToSmiles(mol))
                traceback.print_exc()
                continue
            # Save energies against conformers
            conf_energy_list = []
            for i, e in enumerate(energies):
                conf_energy_list.append([i, e[1]])

            #print(conf_energy_list)

            # Prune again following optmisiation. Start a new mol that will have the conformers removed.
            # When two compounds are within the RMS thresh, pick the one with the lowest energy.
            new_mol = Chem.Mol(mol)

            rms_matrix_list = AllChem.GetConformerRMSMatrix(new_mol, prealigned=False)

            prune_dict = {}
            counter = 0

            # Iterate over the RMS matrix to mark compounds which are below the similarity threshold
            for i in range(len(cids)):
                # Identity is not captured
                if i == 0:
                    continue
                else:
                    for j in range(len(cids)):
                        # Only the bottom half of the matrix is stored
                        if i <= j:
                            continue
                        else:

                            if rms_matrix_list[counter] <= pruneRmsThresh:

                                #logging.debug(f'confs {i} and {j} to be pruned (RMS : {rms_matrix_list[counter]})')

                                # Add both ways to the pruning dictionary
                                if i not in prune_dict.keys():
                                    prune_dict[i] = [j]
                                else:
                                    prune_dict[i].append(j)
                                if j not in prune_dict.keys():
                                    prune_dict[j] = [i]
                                else:
                                    prune_dict[j].append(i)

                            counter += 1    

            # Use the calculated energies to prune the highest energy conformers where needed.

            conf_energy_list.sort(key = lambda x: x[1], reverse = True)

            remove_list = []

            # Remove compounds with a high energy
            if energy_threshold:
                for i, e in conf_energy_list:
                    if e > energy_threshold:
                        remove_list.append(i)

            for i in conf_energy_list:

                conf = i[0]

                # If this is in the dictionary of confs to prune, and hasn't already had all its similars pruned, prune it
                if conf in list(prune_dict.keys()):
                    if len(prune_dict[conf]) == 0:
                        continue
                    else:
                        remove_list.append(conf)
                        del(prune_dict[conf])
                        # If removing it, remove it from all the pruning lists too
                        for conf_comp in prune_dict.keys():
                            if conf in prune_dict[conf_comp]:
                                prune_dict[conf_comp].remove(conf)

            for confId in remove_list:
                new_mol.RemoveConformer(confId)
            # Add pruned mol
            return_mols.append(new_mol)
        
    return return_mols, return_errors

def write_conformers_to_sdf(mol, ID, sdf_file_stream):
    mol.SetProp('_Name', ID)
    for conf in mol.GetConformers():
        sdf_file_stream.write(mol, confId = conf.GetId())
        
def conformer_ensemble_generator(smiles, IDs, num_conf, pruneRmsThresh, energy_threshold, output_file, max_attempts = 1000, additional_optimisation = False):
    
    failed_compounds = []
    sdf_file_stream = Chem.SDWriter(output_file)
    
    for smiles, ID in zip(smiles, IDs):

        mols, failed_compounds = get_conformers_from_smiles(smiles, pruneRmsThresh, num_conf, energy_threshold, max_attempts, additional_optimisation)
        
        for mol in mols:
            write_conformers_to_sdf(mol, ID, sdf_file_stream)

    sdf_file_stream.close()
    
    return failed_compounds
            
def chunk_list(lst, n):
    """Split the list into n chunks.
    Args:
        lst (list): list to split into chunks
        n (integer): number of chunks to make
    Returns:
        list containing chunks of original list"""
    
    return [lst[i::n] for i in range(n)]        
        
def conformer_ensemble_generator_parallel(smiles, IDs, num_conf, pruneRmsThresh, energy_threshold, output_file, max_attempts = 1000, additional_optimisation = False, num_cores = 0):
    
    if num_cores == 0:
        num_cores = len(os.sched_getaffinity(0)) -1
    
    # Chunk the list of molecules
    smiles_chunks = chunk_list(smiles, num_cores)
    IDs_chunks = chunk_list(IDs, num_cores)
    # Run alignment over multiple cores
    
    output_files = [output_file.replace(".sdf", f'{x}.sdf') for x in range(num_cores)]
    
    with multiprocessing.Pool(num_cores) as pool:
        
        # Process each chunk of molecules creating temporary SDF files
        failed_compounds_chunks = pool.starmap(conformer_ensemble_generator, zip(smiles_chunks, IDs_chunks, [num_conf]*num_cores, [pruneRmsThresh]*num_cores, [energy_threshold]*num_cores, output_files, [max_attempts] * num_cores, [additional_optimisation] * num_cores))
    
    # Concatenate the SDF files. Remove the temporary files as they have been processed.
    with open(output_file, "w") as out_file:
        
        for file in output_files:
            
            with open(file, "r") as temp_file:
                for line in temp_file:
                    out_file.write(line)
                    
            os.remove(file)
    
    failed_compounds = []
    
    for failed_compounds_chunk in failed_compounds_chunks:
        failed_compounds.extend(failed_compounds_chunk)
        
    if len(failed_compounds) > 0:
    
        with open(output_file.replace(".sdf", f'_failed.txt'), "w") as error_file:
            
            error_file.write("failed_smiles\n")

            for smiles in failed_compounds:

                error_file.write(f'{smiles}\n')

def main():

    # Parsr arguments

    usage = "Generate conformers using rdkit."

    parser = argparse.ArgumentParser(description = usage,
                                     formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument("--in", 
        action = "store", 
        type = str,
        dest = "input_file",
        help = "Destination of input file containing SMILES and ID",
        required = True)

    parser.add_argument("--out",
        action = "store",
        type = str,
        dest = "output_file",
        help = "Destination of output file",
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
        default = 100)

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
        default = 200)

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