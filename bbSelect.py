#!/usr/bin/env python

"""
Contains the functions to process the bbSelect database to perform a selection

@author: Francesco Rianjongdee
"""

import os
import sys
import logging
import numpy as np
import copy
import math
from bitarray import bitarray
from itertools import chain
import pandas as pd
import time
import multiprocessing
from functools import partial
import traceback
import re
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem, Draw

## All functions creating the query fingerprints are within the classes within this package
import bbSelectPartitioning

class Timer():
    def __init__(self, function = ""):
        self._start = time.perf_counter()
        self._function = function
    def stop(self):
        self._stop = time.perf_counter()
        self._time = self._stop - self._start 
        self._message = f'{self._function} execution time = {round(self._time,2)} seconds'
        logging.info(self._message)

## Functions for loading in the database and querying it

def get_bbgap_configuration(inref):
    """
    Gets configuration of bbGAP database from ref file and returns a dictionary containing the configuration

    Args:

        inref (str): opened reference file
    
    Returns: dictionary containing the configuration
    """
    
    # Start dictionary to contain db config

    DBconfigurationDict = {}

    # Store the pharmacophores used in the fingerprints

    DBconfigurationDict['pharmacophores'] = {}

    # List of paramaters in the configuration.
    # pmap_area is the total 2D area used for pharmacophore capture.
    # fp_size is the length of the fingerprint, including excess bits
    # excess bits is the number of excess bits added to the fingerprint to allow reading as bytes
    # numX is the number of cells along the x-axis
    # numY is the number of cells along the y-axis
    # cell_size is the size of a cell, in squared angstroms

    config_parameters = ['pmap_area','fp_size','excess_bits','numX','numY','cell_size']

    counter = 0

    for line in inref:

        if counter == 0:
            ## Read in the density fingerprint which is the first line in the file and written as a comma separated list of integers
            density_fp = line.strip().strip("[").strip("]").split(", ")
            DBconfigurationDict['density_fp'] = density_fp
            counter += 1

        line = line.strip().split("\t")

        # Iterate over the configuration paramaters and fill in dictionary
        for config in config_parameters:
            if line[0] == config:
                if config == "cell_size":
                    DBconfigurationDict[config] = float(line[1])
                else:
                    DBconfigurationDict[config] = int(float(line[1]))

        # Store pharmacophores
        if line[0] == 'pharmacophore':
            DBconfigurationDict['pharmacophores'][line[1]] = int(line[2])

        # Break when the configuration portion ends
        if line[0] == 'molname':
            break

    # Count number of pharmacophores found
    DBconfigurationDict['num_pharmacophores'] = len(DBconfigurationDict['pharmacophores'].keys())

    return DBconfigurationDict

def byte_to_bit(bytes):
    """
    Convert input bytes to bits.
    Reading the file directly into the bitarray package loses a byte for some reason (bug?)

    Args:
        bytes: Input as bytes
    Returns:
        string : Input converted into bits
    """
    bits = ''.join(format(byte, '08b') for byte in bytes)
    return bits

def count_comma_sep_list(value):
    """
    Get length of comma separated list
    """
    
    return int(len(value.split(",")))

def get_ref_data(inref):
    """
    Gets the compound data from the ref table and calculates the number of unique pharmacophores.
    Args:
        inref (filestream): path to bbgap ref file
    Returns:
        pd.DataFrame of compound data
    """

    inref.seek(0)
    counter = 0
    for line in inref:
        # Count after how many lines the table begins (after configuration lines). 
        # This requires molname to be the first column of the
        line = line.strip().split('\t')
        if line[0] == 'molname':
            break
        else:
            counter +=1

    #logging.debug(f'skip {counter} rows in reference file to get to reference table')
    
    # Read in as csv ignoring the configuration lines
    ref_table = pd.read_csv(inref.name, sep = '\t', skiprows=counter)

    ## Calculate the number of unique pharmacophores and the count of heteroatoms.
    ref_table['count_p'] = ref_table['pharmacophores'].map(count_comma_sep_list).astype(int)

    return ref_table

def create_ref_table(inref, query_fps):
    """
    Creates a pandas dataframe from the database reference file
    Also creates empty columns which will be used to record whether a compound matches any of the pharmacophore queries
    These are in the format [pharmacophore name]_[number of query fingerprint]
    Args:
        inref (filestream): opened reference file
        query_fps (dict): query fingerprint dictionary
    Returns:
        pd.DataFrame of the reference file and query ids
    """
    ## Read in the reference file as a pandas table

    ref_table = get_ref_data(inref)

    # Keep track of column names before the pharmacophore ID columns are added
    ref_table_columns = list(ref_table.columns)
    ref_table_columns.remove('molname')

    ## Add columns for designating matches to query fingerprints

    for pharmacophore in query_fps:
        for i in query_fps[pharmacophore].keys():
            ref_table[pharmacophore+"_"+str(i)] = 0


    # Set the index to molname to speed up searching
    ref_table.set_index('molname',inplace = True)

    return [ref_table,ref_table_columns]

def fp_query(target_fp, query_fps):
    """
    Perform a fingerprint comparison with multiple query fingerprints and a single target fingerprint
    The binary comparison made is checking if there is ANY OVERLAP of bits.
    If there is, it is a match
    
    Args:
        target_fp (bit str) the target fingerprint to compare with the queries
        query_fps (bit str) dictionary containing pharmacophores and the query fingerprints for each pharmacophore
    Returns:
        list of query fingerprint IDs that are matched by the target fingerprint
    """

    pharmacophore_matches = []

    for pharmacophore in query_fps.keys():

        # Iterate over fingerprints for each pharmacophore:

        for fp_number in query_fps[pharmacophore].keys():

            # Fingerprint is held in a the dictionary, keyed by the fingerprint name
            fp = query_fps[pharmacophore][fp_number]

            # Make comparison. If there is a match:
            match = (target_fp & fp).any()

            if match:

                pharmacophore_matches.append(pharmacophore+"_"+str(fp_number))

    return pharmacophore_matches

def parallelize_dataframe(df, func, input_bin, query_fps, fp_bytes, n_cores=5):
    """
    Used to parallelise a function on a dataframe
    
    Args:
        df (pd.DataFrame): array that function will be applied to
        func (function): function to apply to array
        ncores (int): number of cpus to parallelise by
    Returns:
        pd.DataFrame array containing results from function
    """
    input_bin_query_fps_fp_bytes = (input_bin,query_fps,fp_bytes)
    mp = multiprocessing.get_context("spawn")
    pool = mp.Pool(processes = n_cores)
    df_split = np.array_split(df, n_cores)

    df = pd.concat(pool.map(partial(func, input_bin_query_fps_fp_bytes = input_bin_query_fps_fp_bytes), df_split), sort = True)

    pool.close()
    pool.join()
    return df

def get_fp_from_memory_mapped_db(binary_database, start_position, legnth):
    """
    Get a fingerprint from a position and length from a numpy memory mapped array.

    Args:
        binary_database: memory mapped numpy np.ubyte type object
        start_position: start position of fp in bits
        length: length of fingerprint in bits

    Returns: fingerprint in bitarray type object
    """

    # Read the array from the start position with the legnth of desired resulting array
    target_fp = binary_database[int(start_position):int(start_position+legnth)]

    # Convert to a bitarray object
    # Requires the numpy array to be converted to a list to be recognised by the bitarray module

    #target_fp = bitarray(list(target_fp))
    target_fp = bitarray(byte_to_bit(target_fp))

    return target_fp

def process_ref_table_matches(ref_table, input_bin_query_fps_fp_bytes):
    """
    Process the reference table by iterating over the compounds and recording whether they match the query fingerprints

    Args:
        ref_table (pd.DataFrame): pandas DataFrame containing molname, fp_start_position and columns for query fingerprint IDs
        input_bin_query_fps_fp_bytes (list): a list containing the following. Done like this to simplify parallelisation ;)
                         - input_bin: numpy memory mapped uByte array of binary database
                         - query_fps: dictionary containing pharmacophores and the query fingerprints associated with each pharmacophore
                         - fp_bytes: size, in bytes, of fingerprint
    
    Returns:
        pd.DataFrame containing reference dataframe query fingerprint ID matches populated
    """

    # Unpack the packed kwargs
    input_bin, query_fps, fp_bytes = input_bin_query_fps_fp_bytes

    #logging.debug(f'process using {len(ref_table.index)} compounds in chunk')

    # Iterate over compounds in table
    for molname in ref_table.index:
        
        ## Load in the current fp for 'molname'
        ## If the molname occurs multople times, take the minimum fp position.

        byte_position = int(ref_table.loc[molname,'fp_start_position'].min() / 8)

        # Read the fp from the database
        target_fp = get_fp_from_memory_mapped_db(binary_database = input_bin, start_position = byte_position, legnth = fp_bytes)
        
        # Convert fp to a bitarray object. Requires np.Array of bool be converted to list for conversion to bitarray

        # target_fp = bitarray(byte_to_bit(target_fp)) # Conversion happens in function, not using np.Bool types anymore

        # Find matches for fp in query fps

        try:
            matches = fp_query(target_fp = target_fp, query_fps = query_fps)
        except:
            logging.error(f'Problem processing fp for {molname} fp_start-end {byte_position}-{byte_position+fp_bytes} legnth = {fp_bytes}')
            traceback.print_exc()
            sys.exit()

        # Iterate over matches and mark in reference table
        for match in matches:
            ref_table.loc[molname, match] = 1

    return ref_table

def process_db(input_bin, inref, DBconfiguration, query_fps, ncpu):
    """
    Process the database to find matches to the query fingerprints
    Parallelised using python multiprocess
    
    Args:
        inbin (str) binary database file location
        inref (str): opened reference file
        DBconfiguration (dict): Dictionary containing configuration of DB
        query_fps (dict): Dictionary containing query fingerprints
        ncpu (int): Number of cpus to use in parallel

    Returns: 
        ref_table, pd.DataFrame containing all the compounds and which fingerprints they matched
    """

    ref_table,ref_table_columns = create_ref_table(inref = inref, query_fps = query_fps)

    ## Iterate over the fingerprints in the database and capture matches

    # Read the database file as a memory mapped file to make reading more efficient
    # Need to use a numpy memmap boolean array because mmap files are not picklable

    inbin = np.memmap(input_bin,dtype=np.ubyte,mode='r')

    # Determine fp size in bytes

    fp_bytes = int(DBconfiguration['fp_size'] / 8)

    ## Iterate over each of the compounds in the ref_file.
    ## The counter also refers to how many fingerprints have been read
    ## This is done in parallel

    ref_table = parallelize_dataframe(
        df = ref_table, 
        func = process_ref_table_matches, 
        input_bin = inbin, 
        query_fps = query_fps, 
        fp_bytes = fp_bytes, 
        n_cores=ncpu)

    return [ref_table,ref_table_columns]

def SelectCompounds(query_fps, selection_table, selection_dict ,sort_keys,desired_columns, selection_size, DB_configuration, input_bin, tanimoto = 0.99, use_coverage = False, full_coverage = False, ignore_compounds = False, additional_debug = False):
    """
    Determine the selection of compounds based on fingerprint matches and the sort-order

    Args:
        query_fps (dict): Dictionary containing pharmacophores and related fingerprints
        selection_table (pd.DataFrame): Pandas dataframe containing list of compounds, properties and the matches to the fingerprints
        selection_dict (dict): Dictionary keyed by pharmacophore and indicating the number of compounds to choose from each
        sort_keys (list): Comma-separated list of compound properties to sort for selection
        desired_columns (list): Desired columns for the output table

    Returns:
        list containing the table of selected compounds, and a dictionary of the selected compounds, molnames, smiles and matching fp

    """
    # Sort the table by the chosen properties
    selection_table.sort_values(by = sort_keys.split(','), inplace = True)

    if full_coverage:

        selected_smiles = {}
        selected_compounds = []
        # Get the size of the fingeprint in bytes
        fp_bytes = int(DB_configuration['fp_size'] / 8)
        # Open the database file so that fingerprints can be captured
        mmap_db = np.memmap(input_bin, dtype=np.ubyte, mode='r')
        # Set up an empty fingerprint to track the current coverage of the selection
        coverage_fp = bitarray('0'*int(DB_configuration['fp_size']))

        # Iterate through the compounds using the defined prioritisation and select anything that adds to the overall coverage

        for i, row in selection_table.iterrows():

            molname = i
            smiles = row['smiles']
            byte_position = int(row['fp_start_position'] / 8)
            target_fp = get_fp_from_memory_mapped_db(binary_database = mmap_db, start_position = byte_position, legnth = fp_bytes)
            
            # If the compound doesn't add to the overall coveraga
            if coverage_fp == target_fp | coverage_fp:
                continue
            else:
                selected_compounds.append(i)
                selected_smiles[smiles] = {}
                selected_smiles[smiles]['molname'] = molname
                selected_smiles[smiles]['pharmacophore'] = 'Coverage'

                # Update the overall coverage to account for selected compound
                coverage_fp = target_fp | coverage_fp

        selected_compounds = selection_table.loc[selected_compounds]

        return [selected_compounds, selected_smiles]

    # Keep track of excess compounds. These are from partitions that could not be selected from/
    total_excess_compounds = 0

    ## Keep track of the number of matches and the position of selection in the list for all pharmacophore clusters.
    ## List architecture [query_fp_ID, number_matches, selection_position]
    total_count_fp_matches = []

    ## Keep track of IDs to ignore
    ignore_ID = []
    if ignore_compounds:
        ignore_ID = ignore_compounds.split(",")

    ## Keep track of the selected smiles
    selected_rows = []
    selected_smiles = {}

    if use_coverage:
        # Get the size of the fingeprint in bytes
        fp_bytes = int(DB_configuration['fp_size'] / 8)
        # Open the database file so that fingerprints can be captured
        mmap_db = np.memmap(input_bin, dtype=np.ubyte, mode='r')
        # Set up an empty fingerprint to track the current coverage of the selection
        coverage_fp = bitarray('0'*int(DB_configuration['fp_size']))

    # Iterate over the query fingerprints, which are organised by pharmacophore

    for pharmacophore in query_fps.keys():

        ## Keep track of unselected compounds
        excess_compounds = 0

        ## Keep track of the number of matches and selection position for each pharmacophore.
        ## List architecture [query_fp_ID, number_matches, selection_position]
        count_fp_matches = []

        # Keep track of how many compounds are left for selection
        desired_selection = selection_dict[pharmacophore]
        remaining_compounds = copy.deepcopy(desired_selection)

        # Iterate over the query fingerprints for the first time.
        for query_fp_ID in query_fps[pharmacophore].keys():

            # Which row to take from the dataframe, increase this when the compound has already been selected
            selection_position = 0

            ## Find if the selection has already been made, if so go to the next entry in the table
            ## If the selections from picking one per cluster is not good enough
            ## go back to the clusters, arranged by size, and loop over them and continue selecting until you have the correct size
            selected = False

            ## The ID of the query fingerprint in the selection_table 
            query_fp = f'{pharmacophore}_{query_fp_ID}'

            while selected == False:

                ## The proposed selection begins as the first compound in the sorted set of compounds which match the query fingerprint   

                ## If the selection table is empty OR all the compounds have already been selected,
                ## go to the next pharmacophore query fp without removing a compound from the selection count

                if len(selection_table[selection_table[query_fp] == 1].index) <= selection_position:

                    #logging.info(f'{query_fp} - No more compounds available for selection. Will be replaced from another query.')
                    selected = True
                    excess_compounds += 1
                    continue

                else:

                    # Capture number of compounds in cluster
                    number_of_matches = len(selection_table[selection_table[query_fp] == 1].index)

                    # Filter table by compounds which matched the query fingerprint and
                    # Take the compound at the selection position
                    selection = selection_table[selection_table[query_fp] == 1].iloc[[selection_position]]
                    
                    # Trim the selection table to just the desired columns
                    selection = selection[desired_columns]

                    # Extract the smiles, clipped smiles and molname
                    selection_smiles = selection.iloc[[0]]['smiles'].iloc[0]
                    selection_capped_smiles = selection.iloc[[0]]['clipped_smiles'].iloc[0]
                    selection_ID = selection.index

                    ## Determine if the compound should be selected on the basis of options:

                    # Coverage checks that the overall coverage is improved by selecting the compound
                    coverage_pass_fail = 'pass'
                    if use_coverage:

                        # Get the fingerprint for the molecule
                        byte_position = int(selection.iloc[[0]]['fp_start_position'].min() / 8)
                        #print(byte_position)
                        #print(fp_bytes)
                        target_fp = get_fp_from_memory_mapped_db(binary_database = mmap_db, start_position = byte_position, legnth = fp_bytes)

                        # If adding the fp to the already occupied fp doesn't make a difference
                        #print(len(coverage_fp), len(target_fp))
                        if coverage_fp == target_fp | coverage_fp:
                            coverage_pass_fail = 'fail'

                    tanimoto_pass_fail = 'pass'

                    # Tanimoto checks that the compound is below a similarity threshold to be selected
                    if tanimoto > 0:

                        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(selection_smiles), 2, 2048)

                        for smiles in selected_smiles.keys():

                            target_morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)

                            if DataStructs.TanimotoSimilarity(morgan_fp, target_morgan_fp) > tanimoto:
                                tanimoto_pass_fail = 'fail'

                    # Otherwise check if the compound is ignored or whether the compound has already been selected
                    if selection_smiles in selected_smiles.keys() or selection_ID in ignore_ID or tanimoto_pass_fail == 'fail' or coverage_pass_fail == 'fail':
                        if additional_debug:
                            logging.info(f'Molecule {selection_smiles} not selected.')
                            logging.info(f'Molecule already selected: {selection_smiles in selected_smiles.keys()}')
                            logging.info(f'Molecule to be ignored: {selection_ID in ignore_ID}')
                            logging.info(f'Molecule within tanimoto difference: {tanimoto_pass_fail == "fail"}')
                            logging.info(f'Compound does not add to overall coverage: {coverage_pass_fail == "fail"}')

                    # If compound isn't selected, move to the next best compound
                        selection_position += 1
                        continue

                    # Otherwise add the compound to the matches
                    else:

                        # Keep track of the pharmacophore query ID, the number of matches and the current selection position
                        count_fp_matches.append([query_fp,number_of_matches,selection_position])

                        #logging.debug(f'compound {selection_smiles} selected for query fp {query_fp}')

                        ## Keep Track of the query fingerprint the compound was selected for
                        selection['selected_pharmacophore'] = f'{query_fp}'
                        selected_rows.append(selection)

                        # Add smiles to list of selected smiles
                        selected_smiles[selection_smiles] = {}
                        selected_smiles[selection_smiles]['molname'] = selection.index[0]
                        selected_smiles[selection_smiles]['pharmacophore'] = query_fp

                        if use_coverage:
                            # Add fingerprint to overall coverage
                            coverage_fp = target_fp | coverage_fp

                        # Reduce the count of selected compounds and indicate ready to move on to next fingerprint
                        remaining_compounds -=1
                        selected = True

        #logging.info(f'{pharmacophore} - compounds selected  = {desired_selection-remaining_compounds}')
        #logging.info(f'{pharmacophore} - compounds remaining = {remaining_compounds}')

        ## The above loop does not capture the situation where there are too few clusters to satisfy the selection.
        ## This can happen for example if there are two neurons in a single cell in the SOM or if there are empty cluster regions.
        if excess_compounds != remaining_compounds:
            logging.info(f'{pharmacophore} - Too few clusters to satisfy selection for pharmacophore. Reselecting from denser clusters')
            excess_compounds = remaining_compounds
        
        # ------------------------------------------------------------------------------------------------------------------------------
        # The process will now repeat itself to try and make up the desired selection size
        # ------------------------------------------------------------------------------------------------------------------------------

        # Iterate over the populated clusters in descending order of number of compounds
        # This encourages multiple selection from the more densely populated areas in the pharmacophore map

        count_fp_matches.sort(key = lambda x: x[1], reverse = True)
        #logging.debug(f'pharmacophore {pharmacophore} cluster associations: {count_fp_matches}')

        # Set condition that there are still remaning compounds
        no_more_compounds = False

        # Iterate  until there are no more excess compounds for the pharmacophore or until there are no more compounds to select from
        while excess_compounds > 0 and no_more_compounds == False:

            ## If there are no clusters to iterate through:
            if len(count_fp_matches) == 0:
                no_more_compounds = True
                logging.error(f'{pharmacophore} - No more viable compounds to select based on restrictions')
                break

            # Count fp_matches tracked the number of matches for each query in the last round of selection
            # It kept track of the query fp ID, the count of molecules in it and the position of the last selection
            for index,[query_fp,count,selection_position] in enumerate(count_fp_matches):

                # Break out of the for loop (and therefore the while loop) when the selection size has been met
                if excess_compounds <= 0:
                    break

                selected = False

                while selected == False:

                    if len(selection_table[selection_table[query_fp] == 1].index)<=selection_position:

                        # If the table is empty, remove the entry from the list of available clusters
                        try:
                            count_fp_matches.pop(index)
                        except:
                            logging.info(f"No more compounds to select from {query_fp}.")
                            selected = True

                    else:
                        selection = selection_table[selection_table[query_fp] == 1].iloc[[selection_position]]
                        selection = selection[desired_columns]
                        selection_smiles = selection.iloc[[0]]['smiles'].iloc[0]
                        selection_capped_smiles = selection.iloc[[0]]['clipped_smiles'].iloc[0]
                        selection_ID = selection.index

                        # Coverage checks that the overall coverage is improved by selecting the compound
                        coverage_pass_fail = 'pass'
                        if use_coverage:

                            # Get the fingerprint for the molecule
                            byte_position = int(selection.iloc[[0]]['fp_start_position'].min() / 8)
                            target_fp = get_fp_from_memory_mapped_db(binary_database = mmap_db, start_position = byte_position, legnth = fp_bytes)

                            # If adding the fp to the already occupied fp doesn't make a difference
                            if coverage_fp == target_fp | coverage_fp:
                                coverage_pass_fail = 'fail'

                        tanimoto_pass_fail = 'pass'
                        if tanimoto > 0:

                            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(selection_smiles), 2, 2048)

                            for smiles in selected_smiles.keys():

                                target_morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)

                                if DataStructs.TanimotoSimilarity(morgan_fp, target_morgan_fp) > tanimoto:
                                    tanimoto_pass_fail = 'fail'
                        # Otherwise check if the compound is ignored or whether the compound has already been selected
                        if selection_smiles in selected_smiles.keys() or selection_ID in ignore_ID or tanimoto_pass_fail == 'fail' or coverage_pass_fail == 'fail':
                            if additional_debug:
                                logging.info(f'Molecule {selection_smiles} not selected.')
                                logging.info(f'Molecule already selected: {selection_smiles in selected_smiles.keys()}')
                                logging.info(f'Molecule to be ignored: {selection_ID in ignore_ID}')
                                logging.info(f'Molecule within tanimoto difference: {tanimoto_pass_fail == "fail"}')
                                logging.info(f'Compound does not add to overall coverage: {coverage_pass_fail == "fail"}')

                            selection_position += 1
                            continue

                        else:
                            #logging.debug(f'compounds {selection_smiles} selected for query fp {query_fp}')
                            selection['selected_pharmacophore'] = f'{query_fp}'
                            selected_rows.append(selection)
                            # Add smiles to list of selected smiles
                            selected_smiles[selection_smiles] = {}
                            selected_smiles[selection_smiles]['molname'] = selection.index[0]
                            selected_smiles[selection_smiles]['pharmacophore'] = query_fp
                            
                            if use_coverage:
                                # Add fingerprint to overall coverage
                                coverage_fp = target_fp | coverage_fp
                            # reduce the count of selected compounds
                            excess_compounds -=1
                            selected = True

            ## If some compounds haven't been selected for the pharmacophore, add them to excess compounds to take from other pharmacophores
            if excess_compounds > 0:
                logging.info(f'{pharmacophore} - {excess_compounds} selections unable to be performed for pharmacophore. These will be taken from other pharmacophores')
                total_excess_compounds += excess_compounds
        ## Append information on queries which have remaining compounds to the total list
        total_count_fp_matches.extend(count_fp_matches)

    """
    Remaining compounds for selection will now be taken from already explored clusters to satisfy the selection size.
    This is done by iterarating over the clusters from the most populated, and selecting compounds until the deisred selection size is met
    """
    theoretical_remaining_compounds = int(selection_size) - len(selected_rows)
    if theoretical_remaining_compounds != total_excess_compounds:
        total_excess_compounds = copy.deepcopy(theoretical_remaining_compounds)

    if total_excess_compounds > 0:

        ## Capture the number of remaining compounds
        total_remaining_compounds = copy.deepcopy(total_excess_compounds)
        logging.info(f'{total_remaining_compounds} compounds to be selected from non-designated pharmacophores')

        ## Sort the remaining clusters by the number of compounds contained in each

        #logging.debug(f'Final selection round remaining compounds: {total_count_fp_matches}')
        total_count_fp_matches.sort(key = lambda x: x[1], reverse = True)

        while total_remaining_compounds > 0:

            ## If nothing left to select
            if len(total_count_fp_matches) == 0:
                total_remaining_compounds = 0
                logging.error(f'No more compounds can satisfy the selection requirements.')


            for index,[query_fp,count,selection_position] in enumerate(total_count_fp_matches):

                selected = False

                if total_remaining_compounds <= 0:
                    selected = True
                    break

                while selected == False:

                    if total_remaining_compounds <= 0:
                        selected = True
                        break

                    ## If the cluster has been exhausted of compounds
                    if len(selection_table[selection_table[query_fp] == 1].index)<=selection_position:

                        # Remove the entry from the list of available clusters
                        try:
                            count_fp_matches.pop(index)
                            selected = True

                        # If there are no more clusters left, end the cycle
                        except:
                            logging.error(f"No more possible selections can be made under given restrictions.")
                            total_remaining_compounds = 0
                            selected = True
                            break
                    else:
                        selection = selection_table[selection_table[query_fp] == 1].iloc[[selection_position]]
                        selection = selection[desired_columns]
                        selection_smiles = selection.iloc[[0]]['smiles'].iloc[0]
                        selection_capped_smiles = selection.iloc[[0]]['clipped_smiles'].iloc[0]
                        selection_ID = selection.index

                        # Coverage checks that the overall coverage is improved by selecting the compound
                        coverage_pass_fail = 'pass'
                        if use_coverage:

                            # Get the fingerprint for the molecule
                            byte_position = int(selection.iloc[[0]]['fp_start_position'].min() / 8)
                            target_fp = get_fp_from_memory_mapped_db(binary_database = mmap_db, start_position = byte_position, legnth = fp_bytes)

                            # If adding the fp to the already occupied fp doesn't make a difference
                            if coverage_fp == target_fp | coverage_fp:
                                coverage_pass_fail = 'fail'


                        tanimoto_pass_fail = 'pass'
                        if tanimoto > 0:

                            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(selection_smiles), 2, 2048)

                            for smiles in selected_smiles.keys():

                                target_morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 2048)

                                if DataStructs.TanimotoSimilarity(morgan_fp, target_morgan_fp) > tanimoto:
                                    tanimoto_pass_fail = 'fail'

                        # Otherwise check if the compound is ignored or whether the compound has already been selected
                        if selection_smiles in selected_smiles.keys() or selection_ID in ignore_ID or tanimoto_pass_fail == 'fail' or coverage_pass_fail == 'fail':
                            if additional_debug:
                                logging.info(f'Molecule {selection_smiles} not selected.')
                                logging.info(f'Molecule already selected: {selection_smiles in selected_smiles.keys()}')
                                logging.info(f'Molecule to be ignored: {selection_ID in ignore_ID}')
                                logging.info(f'Molecule within tanimoto difference: {tanimoto_pass_fail == "fail"}')
                                logging.info(f'Compound does not add to overall coverage: {coverage_pass_fail == "fail"}')
                            
                            selection_position += 1
                            continue

                        else:
                            #logging.debug(f'compounds {selection_smiles} selected for query fp {query_fp}')
                            selection['selected_pharmacophore'] = f'{query_fp}'
                            selected_rows.append(selection)
                            # Add smiles to list of selected smiles
                            selected_smiles[selection_smiles] = {}
                            selected_smiles[selection_smiles]['molname'] = selection.index[0]
                            selected_smiles[selection_smiles]['pharmacophore'] = query_fp
                            if use_coverage:
                                # Add fingerprint to overall coverage
                                coverage_fp = target_fp | coverage_fp

                            # reduce the count of selected compounds
                            total_remaining_compounds -=1
                            selected = True

    selected_compounds = pd.concat(selected_rows)

    return [selected_compounds, selected_smiles]

def report_text_images(root_name, partitioning, DBconfiguration, clustering_method):
    """
    Creates files to report how the algorithm has run.

    Args:
        root_name (str): The root name used to save all the report files
        partitioning: partitioning object
        DBconfiguration: Dictionary containing the database configuration
        clustering_method: string which clustering method was used
    """

    ## Save information for reporting.

    ## Get date for report
    current_datetime = datetime.now().strftime('%c')

    ## Save the configuration of the pharmacophore map
    fp_size_text = f'Size of fingerprint: {partitioning._full_fp_length} bits\n'
    numy_text = f'Dimension (x,y) of pharmacophore map: ({partitioning._numX},{partitioning._numY})\n'
    try:
        cell_size_text = f'Size of cell edge: {str(float(DBconfiguration["cell_size"]))} Å\n'
    except:
        pass

    ## Save the desired selections text
    selection_from_pharmacophores = partitioning.GetSelection()
    selection_text = "Desired selection from each pharmacophore:\n"
    for i in selection_from_pharmacophores.keys():
        text = f'pharmacophore {i}: {selection_from_pharmacophores[i]}\n'
        selection_text += text

    ## Save visualisations

    #plt.rcParams['figure.figsize'] = [12,8]
    plt.rcParams['figure.figsize'] = [8,6]
    plt.rcParams['font.size'] = 7

    visualisation_name_list = []

    ## Save density map
    density_maps_fig = partitioning.density_heatmaps(trimmed = False)
    density_maps_fig_name = f"{root_name}_density_maps.png"
    density_maps_fig.savefig(density_maps_fig_name)
    visualisation_name_list.append(density_maps_fig_name)

    ## Save the overall clustering for each of the selected pharmacophores
    selection_cluster_maps_fig = partitioning.selection_partition_maps()
    selection_cluster_maps_fig_name = f"{root_name}_selection_cluster_maps.png"
    selection_cluster_maps_fig.savefig(selection_cluster_maps_fig_name)
    visualisation_name_list.append(selection_cluster_maps_fig_name)

    ## Save the coverage visualisation for the selection
    selection_coverage_fig = partitioning.visualise_pharmacophore_coverage(pharmacophore = '*')
    selection_coverage_fig_name = f"{root_name}_coverage_maps.png"
    selection_coverage_fig.savefig(selection_coverage_fig_name)
    visualisation_name_list.append(selection_coverage_fig_name)

    visualisation_dict = {}

    for pharmacophore in partitioning._select_pharmacophores:

        visualisation_dict[pharmacophore] = {}

        if clustering_method == 'som':
            visualisation_dict[pharmacophore]['plot'] = partitioning.pharmacophore(pharmacophore).visualise_som_partitions(annot_density = False, show_neurons = False, ax_alias = pharmacophore)
            cluster_fig_name = f"{root_name}_{pharmacophore}_clusters.png"
            
        elif clustering_method == 'classic':
            visualisation_dict[pharmacophore]['plot'] = partitioning.pharmacophore(pharmacophore).visualise_classic_partitions(annot_density = False)
            cluster_fig_name = f"{root_name}_{pharmacophore}_clusters.png"

        #cluster_fig.Figure(figsize=(10,20))
        visualisation_dict[pharmacophore]['plot'].figure.savefig(cluster_fig_name)
        visualisation_name_list.append(cluster_fig_name)
        plt.clf()

    ## Concatenate text and output to .report file
    report_text = f"bbGAP partitioning report for set {root_name} performed on {current_datetime}\n"
    report_text += f"The pharmacophores were partitioned using the {clustering_method}-based algorithm\n"
    
    if clustering_method == 'som':
        report_text+=f"SOM partitioning parameters:\n"
        report_text+=f"Log transform of densities: {partitioning._log}\n"
        report_text+=f"SOM sigma value: {partitioning._sigma}\n"
        report_text+=f"SOM learning rate: {partitioning._learning_rate}\n"
        report_text+=f"Random seed = {partitioning._seed}\n"
        if partitioning._num_iteration == 'len':
            report_text+=f"SOM number of iterations = length of data\n"
        else:
            report_text+=f"SOM number of iterations = {partitioning._num_iteration}\n"

    report_text += f"{fp_size_text}{numy_text}{cell_size_text}\n"
    report_text += f"{selection_text}\n"

    with open(root_name+".report","w") as output_report:
        output_report.write(report_text)
        output_report.write("Location of visualisation files:\n")
        
        for visualisation in visualisation_name_list:
            output_report.write(visualisation+"\n")

def return_1(value):
    return 1

class Picker():
    """
    Perform all of the process needed to perform a selection
    """
    def __init__(self, ref_file, bin_file, n = 0, method = 'som', pharmacophores = '*', select_mode = 1, ncpu = 20, ignore_compounds = False, sort = None, tanimoto = 0.95, use_coverage = False, flat_som = False, multiplier = 1, **kwargs):
        """
        Upon initialisation of the class, the process will initiate.
        Outputs from the process will be saved as attributes to the class
        Any kwargs given will be passed to the clustering function 
        Args:
            ref_file (str): location of the reference file for the database
            bin_file (str): location of the binary file for the database
            n (int): number of compounds to select
            method (str): between 'som', 'classic', and 'full_coverage'
            pharmacophores (str): comma-separated list of pharmacophores to select from, use * for all
            select_mode (int): determine how many partitions to use for each pharmacophore.
                                    options:0 - proportionally based on total area.
                                                1 - even
                                                2 - proportionally on log scale
                                                3 - proportionally
                                                note - only 1 should be used. Others for testing.
                                                Doing elsewise biases greatly towards hydrophobic
            ncpu (int): number of cpus to distribute the calculations over
            ignore_compounds (str): comma-seperated list of compound IDs to ignore
            sort (str): comma-separated list of values in reference table to prioritise by, in order. Always ascending.
            tanimoto (float): Max tanimoto similarity threshold to enforce between selections
            use_coverage (bool): Whether to enforce that new bits should be set for each selection
            flat_som (bool): Whether the values passed to the SOM should contain 1 in each occupied cell
            multiplier (float): Multiply the number of values passed to the SOM to provide additional training data.
        """

        # Whether to pass all values to the SOM as 1, flattening the disribution so performing something analogous to the classic partitioning

        if flat_som == True:
            transform_func = return_1
        else:
            transform_func = None

        if sort == None:
            raise ValueError('Must define a method to prioritise selections in the "sort" argument')
        ## Open reference file
        inref = open(ref_file, "r")

        ## Get the configuration
        self._DBconfiguration = get_bbgap_configuration(inref)

        #logging.debug(f'Configuration:\n{self._DBconfiguration}')

        ## Create query fingerprints

        # Initiate the clustering class which performs all of the analysis and clustering

        logging.info("Beginning generation of query fingerprints")
        query_fps_timer = Timer("Generation of query fingerprints")

        self._Partitioning = bbSelectPartitioning.selection_maps(
                                fp = self._DBconfiguration['density_fp'],
                                numX = self._DBconfiguration['numX'],
                                numY = self._DBconfiguration['numY'],
                                pharmacophore_dict = self._DBconfiguration['pharmacophores'],
                                excess_bits = self._DBconfiguration['excess_bits'])

        # Set the selection parameters and calculate how many compounds to take from each pharmacophore
        self._Partitioning.SetSelection(selection = n, p_select = pharmacophores, select_mode = select_mode)

        # Get the number of compounds to select from each pharmacophore
        self._selection_dict = self._Partitioning.GetSelection()

        # Set and perform the method of paritiioning to use

        if method in ['som', 'classic']:
            full_coverage = False
            if method == 'som':
                self._Partitioning.som_partitioning(transform_func = transform_func, multiplier = multiplier, **kwargs)
            elif method == 'classic':
                self._Partitioning.classic_partitioning(**kwargs)

            # Retrieve the query fingerprints
            self._query_fps = self._Partitioning.GetQueryFps()

            query_fps_timer.stop()

            ## Process fingerprint database to find matches to query fingerprints
            ## This takes the ref file and the fingerprint binary database and categorises each compound depending on which fingerprint it matches
            ## This is performed over multiple cores as it is the most computationally expensive step, as every compound is compared to every query fp
            
            logging.info(f'Starting fingerprint matching algorithm over {ncpu} cores')

            process_db_timer = Timer("fingerprint matching")

            self._selection_table,self._ref_table_columns = process_db(input_bin = bin_file, 
                                        inref = inref, 
                                        DBconfiguration = self._DBconfiguration, 
                                        query_fps = self._query_fps,
                                        ncpu = ncpu)
            process_db_timer.stop()

        elif method == 'full_coverage':
            full_coverage = True
            self._selection_table = get_ref_data(inref)
            self._selection_table.set_index('molname',inplace = True)
            self._ref_table_columns = list(self._selection_table.columns)
            self._Partitioning.full_coverage()
            self._query_fps = {}

        else:

            raise ValueError(f'method {method} is invalid. Choose between "som", "classic", and "full_coverage"')

        self._method = method

        self._coverage = self._Partitioning._coverage 
        ## Selection of compounds
        ## This goes through the selection table and decides which compounds to select.

        logging.info('Starting selection algorithm')

        selection_timer = Timer("selection")

        self._selected_compounds, self._selected_smiles = SelectCompounds(
                                                    query_fps = self._query_fps, 
                                                    selection_table = self._selection_table, 
                                                    selection_dict = self._selection_dict,
                                                    sort_keys = sort,
                                                    desired_columns = self._ref_table_columns,
                                                    ignore_compounds = ignore_compounds,
                                                    selection_size = n,
                                                    DB_configuration = self._DBconfiguration,
                                                    input_bin = bin_file,
                                                    tanimoto = tanimoto,
                                                    use_coverage = use_coverage,
                                                    full_coverage = full_coverage)
        selection_timer.stop()

        inref.close()

        ## Prepare coverage
        self._Partitioning.Coverage(bin_loc = bin_file, ref_loc = ref_file)
        self._Partitioning.enumerate_coverage(molnames = self.GetSelectionIDs())

    def VisualisePartitioning(self, pharmacophore = None, figsize = [15,10]):
        """
        Return Seaborn figure object containing the clusters visualisation
        """
        plt.rcParams['figure.figsize'] = figsize

        #return self._Partitioning.selection_partition_maps()
        self._Partitioning.selection_partition_maps()

    def GetSelectionSMILES(self):
        """
        Return list of smiles selected
        """
        return list(self._selected_smiles.keys())

    def GetSelectionDict(self):
        """
        Return dictionary containing the calculated selection keyed by smiles and containing the name of the pharmacophore they were selected for
        """
        return self._selected_smiles

    def GetSelectionIDs(self):
        """
        Return list of IDs selected
        """
        return [self._selected_smiles[x]['molname'] for x in self._selected_smiles.keys()]

    def SaveSelection(self, **kwargs):
        """
        Saves the selection as a csv (or other) file
        :param location: file location to save to
        :param delimiter: delimiter
        """
        self._selected_compounds.to_csv(**kwargs)

    def SaveClusters(self, **kwargs):
        """
        Saves the clustered unput as a csv (or other) file
        :param location: file location to save to
        :param delimiter: delimiter
        """
        self._selection_table.to_csv(**kwargs)

    def VisualiseCoverage(self, pharmacophore = '*', figsize = [15,10]):
        """
        Visualises the coverage of the chemical space made by the selection
        :param pharmacophore: which pharmacophore to return. Use * for all.
        """
        plt.rcParams['figure.figsize'] = figsize
        #plt.figure(figsize=figsize)
        return self._Partitioning.visualise_pharmacophore_coverage(pharmacophore = pharmacophore)

    def GetAllSmiles(self):
        return list(self._selection_table['smiles'])

    def GetAllIds(self):
        return list(self._selection_table.index)

    def GetDataTable(self):
        return self._selection_table

    def DrawSelectedMols(self, align_smiles, fontsize = 12, molsPerRow = 8, legendFraction = 0.25):
        """
        Draw selected mols from bbSelect selected dictionary
        """

        opts = Draw.MolDrawOptions()
        #opts.legendFraction = 0.25
        opts.legendFontSize = fontsize
        opts.legendFraction = legendFraction

        # Align all the attachment points together
        align_mol = Chem.MolFromSmiles(align_smiles)
        AllChem.Compute2DCoords(align_mol)

        selected_smiles_dict = self.GetSelectionDict()
        
        selected_mols = []
        for smiles in selected_smiles_dict.keys():
            try:
                selected_mol = Chem.MolFromSmiles(smiles)
            except:
                print(f'could not convert {smiles} to mol')
                
            AllChem.GenerateDepictionMatching2DStructure(selected_mol , align_mol)
            selected_mol.SetProp('_Name',f"{selected_smiles_dict[smiles]['pharmacophore']}")
            selected_mols.append(selected_mol)

        return Draw.MolsToGridImage(selected_mols, molsPerRow= molsPerRow ,legends = [x.GetProp("_Name") for x in selected_mols],subImgSize=(150,150), maxMols = np.inf, useSVG=True, drawOptions = opts)

    def SimulateCoverage(self, molnames):
        """
        Supply molnames to visualise their coverage.
        """

        coverage = copy.deepcopy(self._Partitioning)
        coverage.enumerate_coverage(molnames = molnames)
        fig = coverage.visualise_pharmacophore_coverage(pharmacophore = '*')

        return coverage._coverage

    def VisualisePartitioningDetail(self, pharmacophore, figsize = [15,10], **kwargs):
        """
        Visualise partitions for a single pharmacophore with the partition names
        """
        plt.rcParams['figure.figsize'] = figsize
        if self._method == 'som':
            self._Partitioning.pharmacophore(pharmacophore).visualise_SOM_partitions(show_neurons = True, untrimmed = False)

        elif self._method == 'classic':
            self._Partitioning.pharmacophore(pharmacophore).visualise_classic_partitions(annot_density = False)

    def VisualisePropertyDistribution(self, override_selected_IDs = None, twinx = True):
        """
        Visualise property distributions of selected set compared to total set
        Args:
            override_selected_IDs (list or None): list of IDs to use if not the selected ones
            twinx (bool): Whether to share x-axis for references
        """

        if override_selected_IDs != None:
            selected_IDs = override_selected_IDs
        else:
            selected_IDs = self.GetSelectionIDs()

        return visualise_properties(table = self.GetDataTable(), 
            selected_molnames = selected_IDs, 
            reference_molnames = self.GetAllIds(), 
            twinx = twinx)

def visualise_properties(table, selected_molnames, reference_molnames = False, twinx = False):
    """
    This generates plots used to compare property distributions in the example notebooks.
    Args:
        table (pd.DataFrame): table containing all the date
        selected_molnames (list): IDs that are selected
        reference_molnames (list): IDs to compare to
        twinx (bool): Whether to share x-axis for references.
    """
    f, (axes) = plt.subplots(nrows = 3, ncols = 2, sharey = False, sharex = False, figsize = (12,10))
    properties = {
        'heavy': {
            'range': (0,12),
            'bins' : 12,
            'alias': 'heavy atoms'},
        'mw': {
            'range': (0,160),
            'bins' : 16,
            'alias' : 'molecular weight'},
        'rb': {
            'range': (0,6),
            'bins' : 6,
            'alias' : 'rotatable bonds'},
        'het_count': {
            'range': (0,7),
            'bins' : 7,
            'alias' : 'heteroatom count'},
        'chiral': {
            'range': (0,5),
            'bins' : 5,
            'alias' : 'chiral centres'},
        'count_p': {
            'range': (0,6),
            'bins' : 6,
            'alias' : 'unique pharmacophores'},
    }

    for i,(m_property, ax) in enumerate(zip(properties.keys(), chain.from_iterable(axes))):
        mean_value = round(table[table.index.isin(selected_molnames)][[m_property]].mean()[0],1)
        if reference_molnames:
            if twinx:
                ax2 = ax.twinx()
                ax2.set_ylabel('Total set count')
            else:
                ax2 = ax
            ax2.hist(table[table.index.isin(reference_molnames)][[m_property]], range = properties[m_property]['range'], bins = properties[m_property]['bins'], color = 'grey', alpha = 0.3, hatch='/', edgecolor='grey')
        ax.hist(table[table.index.isin(selected_molnames)][[m_property]], range = properties[m_property]['range'], bins = properties[m_property]['bins'], color = plt.get_cmap('tab20')((i/6)-0.05), alpha = 0.7)        
        ax.set_title(f"{properties[m_property]['alias']} - {mean_value}")
        ax.set_ylabel('Selected set count')
    f.tight_layout()

    return f

def main():

## Parse arguments

    usage = "Perform selection from bbSelect database."

    parser = argparse.ArgumentParser(description = usage,
                                     formatter_class=argparse.RawTextHelpFormatter,)
    parser.set_defaults(ref = False,
                       log = False,
                       verbose = False,
                       n_select = 24,
                       p_select = '*',
                       keep = False,
                       positive_x = False,
                       debug = False,
                       sort = None,
                       ncpu = 5,
                       subarea_ratio = 0.6,
                       big_x = True,
                       over_select = False,
                       method = 'som',
                       output_file = False,
                       select_mode = 2,
                       ignore_compounds = False,
                       report = False)

    parser.add_argument("input", metavar = "INPUT_FILE", 
                     help = "path of input bbSelect database. May be .bin, .ref or root name\n")

    parser.add_argument("-s","--select",action = "store",type = str,dest = "n_select",
                     help = "The size of the selection, integer."+
                     "Default: %(default)s \n")

    parser.add_argument("-p","--pharmacophores",action = "store",type = str,dest = "p_select",
                     help = "comma-separated list of pharmacophores to select from.\n"+
                     "Use * to denote all pharmacophores. or e.g. '01_A,04_D,09_R'\n"
                     "Use : to assign a number to select from pharmacophores. Overrides other selection settings'\n"
                     "Default: %(default)s \n")

    parser.add_argument("-e","--mode",action = "store",type = int,dest = "select_mode",
                     help = "Indicate how the selection should be partitioned between pharmacophores.\n"+
                     "1 = evenly, 3 = completely proportionally, 2 = intermediate (recommended)\n"
                     "Default: %(default)s \n")

    parser.add_argument("-m","--method",action = "store",type = str, dest = "method",
                     help = "Which clustering method to use.\n"+
                     "Available methods:\n \t'som' - Use of SOM to cluster based on the make-up of the set\n"+
                     "\t'classic' - Evenly divide area of pharmacophore map.\n"+
                     "\t'full_coverage' - Select compound until full coverage has been achieved.\n"+
                     "Default: %(default)s \n")

    parser.add_argument("-n","--ncpu",action = "store",type = int,dest = "ncpu",
                     help = "\nHow many cpus to use. Default: %(default)d \n")

    parser.add_argument("-b","--sort",action = "store",type = str,dest = "sort",
                     help = "\nWhich values to use to sort results. Values from cprops file, comma separated \n"+
                     "Default: %(default)s \n")

    parser.add_argument("-o","--out",action = "store",type = str, dest = "output_file",
                     help = "\nRoot name of output file. Default: stdout for smi, others will inherit input file name\n"+
                     "Will create three files. [name]_selection.smi, [name]._selection.tsv and [name].clustered.tsv\n")

    parser.add_argument("-v","--verbose",action = "store_true",dest = "verbose",
                     help = "\nApplication runs verbosely\n"+
                     "use in tandem with --log to store the logs in a file. Default: %(default)s \n")

    parser.add_argument("-d","--debug",action = "store_true",dest = "debug",
                     help = "\nApplication runs in degug logging mode\n"+
                     "use in tandem with --log to store the logs in a file. Default: %(default)s \n")

    parser.add_argument("-i","--ignore_compounds",action = "store", type = str, dest = "ignore_compounds",
                     help = "\nprovide comma-separated list of compound IDs to ignore in selection\n"+
                     "Default: %(default)s \n")

    parser.add_argument("-r","--report",action = "store_true", dest = "report",
                     help = "\nOutputs visualisations of the clustering process\n"+
                     "Default: %(default)s \n")

    options = parser.parse_args()

    ## Check arguments and options are fine

    # Allow use to use .ref or .bin file as input

    extensions = ['.ref','.bin']
    root_name = options.input

    for extension in extensions:
        if extension in options.input:
            root_name = options.input.replace(extension, "")

    # Put the cwd into the root name if a full path wasn't provided
    if root_name.startswith("/"):
        pass
    else:
        if root_name.startswith("./"):
            root_name.replace("./",os.getcwd())
        else:
            root_name = os.getcwd()+"/"+root_name
    
    # Check if the input files exist:

    input_names = [root_name+".bin",root_name+".ref"]   

    for name in input_names:

        if not os.access(name,os.R_OK) or not os.path.isfile(name):
            logging.error(f'Input file {name} does not exist. Please use an appropriate root name')
            sys.exit(2)

    input_bin, input_ref = input_names


    # Check if integer or string provided in selection option by trying to convert it.
    """try: 
        selection = int(options.n_select)
    except:
        selection = options.n_select
        pass"""

    # Check logs and verbose:
    verbose = options.verbose

    if options.output_file:
        root_name = options.output_file

    if options.log:
        verbose = True

    if options.method not in ['som','classic']:
        logging.error('Method must be either "som" or "classic".')
        sys.exit(2)

    # Configure logger depending on logfile or verbose:

    if options.log:
        logging.basicConfig(format = '%(levelname)s -%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.INFO)
    elif options.verbose:
        logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.INFO)
    elif options.debug:
        logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.DEBUG)
    else:
        logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.ERROR)

## Main code

    ## Initiate picker class

    bbSelection = Picker(
                        ref_file = input_ref, 
                        bin_file = input_bin, 
                               n = options.n_select,  # How many compounds to select
                          method = options.method, # Which clustering method to use
                  pharmacophores = options.p_select, # Which pharmacophore to select from
                     select_mode = options.select_mode, # How the selection size is divided between the pharmacophores
                            ncpu = options.ncpu,  # Number of cpus to use
                ignore_compounds = options.ignore_compounds, # Which compounds to ignore, can be a list
                            sort = options.sort) # Which properties to sort on

    ## Save the cluster associations with each compound
    bbSelection.SaveClusters(path_or_buf = root_name+'_clustered.csv', sep = ",", index = True)

    ## Save the selection as a csv
    bbSelection.SaveSelection(path_or_buf = root_name+'_selected.csv', sep = ",", index = True)

    ## This section is for saving information for reporting.
    if options.report:
        bbSelection._Partitioning.Coverage(bin_loc = input_bin, ref_loc = input_ref)
        selected_molnames = bbSelection.GetSelectionIDs()
        bbSelection._Partitioning.enumerate_coverage(molnames = selected_molnames)
        report_text_images(root_name = root_name, partitioning = bbSelection._Partitioning, DBconfiguration =bbSelection._DBconfiguration, clustering_method = options.method)

if __name__ == '__main__':
    main()