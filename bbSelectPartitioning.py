#!/usr/bin/env python

"""
Contains the classes and functions required to peform the partitioning of the bbGAP space.

@author: Francesco Rianjongdee

"""

import sys
import pandas as pd
try:
    import seaborn as sns
except:
    pass
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import logging
import traceback
import math
import random
import traceback
from itertools import chain
from bitarray import bitarray
from minisom import MiniSom

## For visualisation
import matplotlib.colors as colors
import matplotlib.cm as cmx

## For rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import Draw, PandasTools

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

plt.rcParams['figure.figsize'] = [15,10]

class density_map():
    """
    This class is set up for each pharmacophore and contains all information and processed maps (density map)
    Upon initiation with the occupancy fingerprint
    """
    def __init__(self, pharmacophore, pharmacophore_posn, full_fp_length, pharmacophore_density_fp, untrimmed_number_map):
        """
        Args:
            pharmacophore (str): name of pharmacophore
            pharmacophore_posn (int): position of pharmacophore on fingerprint
            full_fp_length (int): length of fingerprint
            pharmacophore_density_fp (list): The overall fingerprint occupancy for that pharmacophore
            untrimmed_number_map (np.Array): Array showing the fingerprint position for each of the cells.
        """
        self._pharmacophore = pharmacophore
        self._density_fp = pharmacophore_density_fp
        self._untrimmed_number_map = untrimmed_number_map

        # Get the number of X and Y cells from dimensions of the number map
        self._numY, self._numX = self._untrimmed_number_map.shape

        # Create the selection fingerprint, which is empty for now
        self._selection_fp = [0] * self._numY * self._numX

        # Create dictionary which will contain SOM query fps:
        self._set_driven_clustering_fps = {}

        # Create dictionary which will contain the even-division query fps:
        self._space_driven_clustering_fps = {}

        # Create density map from fp
        self.enumerate_untrimmed_density_map()

        # Create trimmed maps
        self._density_map, self._number_map = TrimMap(
            density_map = self._untrimmed_density_map, 
            number_map = self._untrimmed_number_map)

    ## These functions do calculations

    def enumerate_untrimmed_density_map(self):
        """
        Create untrimmed density map from numY, numX and density fp by creating a np.Array with the overall occupancy related to the cells
        """
        ## Create untrimmed density map, which is an empty numpy array
        self._untrimmed_density_map = np.ones((self._numY, self._numX))

        # Enumerate the density map
        counter = 0
        for y in range(self._numY):
            for x in range(self._numX):
                try:
                    self._untrimmed_density_map[y,x] = self._density_fp[counter]
                except:
                    logging.error(f'Error enumerating untrimmed map for  pharmacophore {self._pharmacophore} position {x,y} from density fingerprint position {counter}')
                counter += 1

    def SOM_partition(self, x, y, log = 2, multiplier = 1, sigma = 1, learning_rate = 0.5, num_iteration = 'len', topology = 'rectangular', random_weights_init = True, neighborhood_function = 'gaussian', activation_distance = 'euclidean', seed = 3, transform_func = None):
        """
        Perform SOM clustering on the density map object (np.Array)
        This will convert the array into discrete data for the SOM, train the SOM and then convert the output to a partitioning.

        Arguments have reflect those in the minisom package
        See the train_SOM() function for argument information        
        """
        # Convert the array of cells to discrete values
        self._density_scatter_table = discretise_data(self._density_map, log = log, random = True, multiplier = multiplier, seed = seed, transform_func = transform_func)
        self._density_scatter_values = self._density_scatter_table.values

        # Train a som on the data
        self._som = train_SOM(data = self._density_scatter_values, x=x, y=y, 
                      sigma = sigma, 
                      learning_rate = learning_rate, 
                      num_iteration = num_iteration, 
                      topology = topology, 
                      random_weights_init = random_weights_init,
                      neighborhood_function = neighborhood_function,
                      activation_distance = activation_distance,
                      seed = seed)

        # Determine the partitions from the SOM, get the neuron coordinates and the visualisations
        self._SOM_partitions = SOM_partitions(som = self._som, data = self._density_scatter_values)
        self._neuron_coordinates = getNeuronCoordinates(self._som,x,y)
        self._SOM_partition_map, self._SOM_partition_map_string, self._SOM_partition_dict = clusterMapFromSOM(self._density_map, self._SOM_partitions, random_sort = True)
        
        # Create the fingerprints to use to match compounds to their partitions
        self.SOM_partition_selection_fps()

    def SOM_partition_selection_fps(self):
        """
        Create the fingerprints to use to match compounds to their partitions
        specific for the pharmacophore's region of the fp
        """

        # Get a list of unique clusters from the SOM output
        clusters = np.unique(self._SOM_partition_map_string[self._SOM_partition_map != 0])

        # Iterating over each of the clusters:
        for cluster in clusters:

            # Get the coordinates where those clusters exist
            cluster_coords = np.where(self._SOM_partition_map_string == cluster)

            # Get the relevant cell numbers (i.e. the fingerprint positions)
            cluster_cell_numbers = self._number_map[cluster_coords]

            # Take a copy of a blank fingerprint to manipulate to be the query fp
            query_fp = cp.deepcopy(self._selection_fp)

            # Turn on the bits at the representative parts of the fingerprint
            for cell_number in cluster_cell_numbers:
                query_fp[int(cell_number)] = 1

            # Add query fp to dictionary, keyed by cluster name
            self._set_driven_clustering_fps[cluster] = query_fp

    def analyse_SOM_partitions(self, visualise = True):
        """
        Used to create a pandas dataframe with the analysis of partitioning composition.
        This contains the number of pharmacophore feature placements in each partition and the distribution of partition areas
        """
        self._SOM_partition_analysis_dict = analyse_clusters(cluster_map = self._SOM_partition_map_string, density_map = self._density_map)
        analysis_table = pd.DataFrame.from_dict(self._SOM_partition_analysis_dict, orient = 'index')
        analysis_table.index.rename('cluster')

        if visualise:
            analysis_table.plot.bar(subplots = True)

        return analysis_table

    def analyse_classic_partitions(self, visualise = True):
        """
        Used to create a pandas dataframe with the analysis of partitioning composition.
        This contains the number of pharmacophore feature placements in each partition and the distribution of partition areas
        """
        self._classic_partition_analysis_dict = analyse_clusters(cluster_map = self._classic_partition_map_string, density_map = self._density_map)
        analysis_table = pd.DataFrame.from_dict(self._classic_partition_analysis_dict, orient = 'index')
        analysis_table.index.rename('cluster')

        if visualise:
            analysis_table.plot.bar(subplots = True)

        return analysis_table

    ## These functions to visualisations

    def visualise_som(self):
        """
        Visualise the discrete data that was used to train the SOM as a scatter plot
        Each point is coloured by the closest SOM neuron
        Overlaid are the positions of the SOM neurons
        """

        visualise_som_scatter(
            SOM_partition_dataframe = self._SOM_partitions, 
            neuron_map = self._neuron_coordinates, 
            random_order = True)

    def visualise_SOM_partitions(self, annot_density = False, show_neurons = False, ax_alias = False, untrimmed = False):
        """
        Visualise the cells that have been partitioned together by the SOM
        The cells can be annotated either by the density in the cell or by the cell name
        Uses the seaborn heatmap visualisation, and assigns a colour to each cluster
        
        Args:
            annot_density (bool): set to True to annotate the cells with the density
            show_neurons (bool): show the positions of the Neurons
            ax_alias (bool or dict): use to stop bug where plots are overlain on each other. Set to false if to run normally, set to dictionary variable e.g. dict[pharmacophore] to allow a different alias for each ax instance made.
            untrimmed (bool): whether to trim the unoccupied cells if they form a border around the visualisation.
        Returns: plt.figure

        """
        if untrimmed:
            SOM_partition_map = untrim_array(
                        self._SOM_partition_map, 
                        self._number_map,
                        self._untrimmed_number_map,
                        string_type = False)
            SOM_partition_map_string = untrim_array(
                        self._SOM_partition_map_string, 
                        self._number_map,
                        self._untrimmed_number_map,
                        string_type = True)
            density_map = self._untrimmed_density_map

        else:
            SOM_partition_map = self._SOM_partition_map
            SOM_partition_map_string = self._SOM_partition_map_string
            density_map = self._density_map

        fig = visualise_SOM_partitions(
            SOM_partition_map = SOM_partition_map,
            SOM_partition_map_string = SOM_partition_map_string,
            density_map = density_map,
            annot_density = annot_density, 
            neuron_coordinates = self._neuron_coordinates, 
            show_neuron_coordinates = show_neurons,
            title = self._pharmacophore,
            ax_alias = ax_alias)

        return fig

    def visualise_scatter(self):
        """
        Visualise the scatter plot created for the SOM
        """
        self._density_scatter_table.plot.scatter(0,1)

    def density_heatmap(self, untrimmed = False, new_style = True, square = False):
        """
        Plot heatmap of densities on x,y plot
        """
        if untrimmed:
            array = self._untrimmed_density_map
        else:
            array = self.density_map()
            
        if new_style:
            zeros = np.zeros(array.shape)
            zeros = np.where(array <1, 0, np.NaN)
            array = np.where(array > 0, array, np.NaN)
            square = True
            sns.heatmap(zeros, annot = True, fmt = 'g', cmap = 'Greys', robust = True, cbar = False, square = square)


        sns.heatmap(array, annot = True, fmt = 'g', cmap = 'coolwarm', robust = True, cbar = False, square = square)
        plt.gca().invert_yaxis()

    def visualise_classic_partitions(self, annot_density = True, cmap = 'Spectral'):
        """
        Visualise the cells that have been clustered together using classic partitioning
        The cells will be annotated by their density
        Uses seaborn heatmap visualisation, assigning a colour to each cluster
        
        Args:
            annoy_density (bool): annotate the cells with the density (total occupancy)
            cmap (str): matplotlib cmap colour scheme
        """

        array = self._classic_partition_map_string
         
        array = np.where(self._density_map == 0, np.NaN, array)


        if annot_density:
            annot = self._density_map
            fmt = 'g'
        else:
            annot = self._classic_partition_map_string
            fmt = 's'

        clusters = np.unique(self._classic_partition_map_string)

        ## Randomise clusters so that the colours have more contrast
        clusters = random.sample(sorted(clusters),len(clusters))

        counter = 1

        array_as_numbers = np.zeros((array.shape))

        for cluster in clusters:
            coords = np.where(array == cluster)
            array_as_numbers[coords] = counter
            counter += 1

        # get rid of zero areas

        array_as_numbers[np.where(self._density_map == 0)] = np.NaN

        ax = plt.axes()

        sns.heatmap(array_as_numbers, cmap = cmap, cbar = False, square = True, robust = True, annot = annot, fmt = fmt,ax = ax)

        plt.gca().invert_yaxis()

        plt.title(self._pharmacophore)

        return ax

    ## These functions perform the clustering from a selection size

    def som_partitioning(self, selection_size, log = 2, multiplier = 1, sigma = 1, learning_rate = 0.5, num_iteration = 'len', topology = 'rectangular', random_weights_init = True, neighborhood_function = 'gaussian', activation_distance = 'euclidean', big_x = True, seed = 3, transform_func = None):
        """
        Perform partitioning of the bbGAP space using self-organising maps 
        Here the clusters are reflective of the distribution of pharmacophore features within the set
        Smaller partitions are formed in highly occupied regions of chemical space
        Larger artitions are formed in sparsely occupied regions of chemical space

        The size of the SOM is determined from the selection size by picking the most 'square' dimensions possible.
        See the train_SOM() function for detail about arguments

        Args:        
            selection_size (int): size of selection, also equivalent to number of neurons in SOM 
                
        """

        # Get potential integer factors for the selection size
        potential_SOM_dimensions = find_available_dimensions(selection_size)

        # Iterate over these and calcualate the difference between the dimensions
        # add these as a third item

        for i, dimensions in enumerate(potential_SOM_dimensions):
            dimension_difference = abs(dimensions[0]-dimensions[1])
            potential_SOM_dimensions[i].append(dimension_difference)

        # Sort the dimensions by the differences in the dimensions
        potential_SOM_dimensions = sorted(potential_SOM_dimensions, key = lambda x: x[2])

        # Select the dimensions with the smallest difference
        chosen_dimensions = potential_SOM_dimensions[0][:2]

        # Set these as the chosen x and y coordinates
        x,y = sorted(chosen_dimensions, reverse = True)

        self.SOM_partition(x, y, log, multiplier, sigma, learning_rate, num_iteration, topology, random_weights_init, neighborhood_function, activation_distance, seed = seed, transform_func = transform_func)

    def classic_partitioning(self, selection_size, dimension_ratio = 0.6, trimmed = True, method = 'round'):
        """
        Perform the classic partitioning of the space and create the query fingerprint to match compounds to their partitions

        Args:
            selection_size (int): the number of compounds to select
            dimension_ratio (float): the minimum ration of short:long edge of selection area
            method (str): The method to use to convert from float to int. Options: 'round', 'ceil', 'floor'
            trimmed (bool): whether to trim the empty rows and columns from the array
        """

        ## Start a dictionary for the fingerprint

        self._space_driven_clustering_fps = {}

        ## Whether to use trimmed or untrimmed density maps

        if trimmed:
            array = self._number_map
        else:
            array = self._untrimmed_number_map

        ## Create divided array
        self._classic_partition_map_string, chosen_selection_size = partition_array(array, selection_size, dimension_ratio, method = 'round')

        ## Notify if selection size changed
        if chosen_selection_size < selection_size:
            logging.warning(f'selection size for pharmacophore {self._pharmacophore}: {chosen_selection_size}\nThis is less than requested as the space could not be divided by selection size\nThe remaining compounds will be taken from the more populated regions')

        cluster_IDs = np.unique(self._classic_partition_map_string)

        # Iterating over each of the clusters:
        for cluster in cluster_IDs:

            # Get the coordinates where those clusters exist
            cluster_coords = np.where(self._classic_partition_map_string == cluster)

            # Get the relevant cell numbers (i.e. the fingerprint positions)
            cluster_cell_numbers = self._number_map[cluster_coords]

            # Take a copy of a blank fingerprint to manipulate to be the query fp
            query_fp = cp.deepcopy(self._selection_fp)

            # Turn on the bits at the representative parts of the fingerprint
            for cell_number in cluster_cell_numbers:
                query_fp[int(cell_number)] = 1

            # Add query fp to dictionary, keyed by cluster name
            self._space_driven_clustering_fps[cluster] = query_fp

    ## These functions return values of the object

    def query_fps(self):
        """
        Returns query fps
        """
        return self._query_fps

    def untrimmed_number_map(self):
        """
        Returns untrimmed number map
        """
        return self._untrimmed_number_map

    def number_map(self):
        """
        Returns trimmed number map (default, no need to use untrimmed really)
        """
        return self._number_map

    def untrimmed_density_map(self):
        """
        returns untrimmed density map
        """
        return self._untrimmed_density_map

    def density_map(self):
        """
        returns untrimmed density map
        """
        return self._density_map

class selection_maps():
    """
    This class takes the entire occupancy count list (count of on bits at each position of a fingerprint)
    It then processes the list for each of the contained pharmacophores
    This performs the partitioning across the pharmacophores, and generate the visualisations
    
    """
    def __init__(self, fp, numY, numX, pharmacophore_dict, excess_bits, seed = 3):
        """
        Args:
            fp (list): The overall occupancy count list
            numY (float): Number of cells on the y-axis
            numX (float): Number of cells on the x-axis
            pharmacophore_dict (dict): Dictionary detailing the order of the pharmacophores in the fingerprint
            excess_bits (int): The number of excess bits used when creating the fingerprint
            seed (float): the random seed to use.
        """

        ## Get the density fingerprint (counts of 1's at each position in the fingerprints for the database)
        self._fp = fp


        # The number of X and Y cells
        self._numY = numY
        self._numX = numX

        # A dictionary keyed by the different available pharmacophores and containing which position (order) the pharmacophore lies in the fingerprint
        self._pharmacophore_dict = pharmacophore_dict

        # Get the number of excess bits
        self._excess_bits = excess_bits

        # The size of the fingerprint / area of the pharmacophore map
        self._fp_area = numY * numX

        # Legnth of total fp (without excess bits i think)
        self._full_fp_length = len(self._fp)

        ## Create empty maps which are the maps containing densities and maps containing cell number.
        ## These are used as reference for the pharmacophore density maps, the trimmed maps and the clustered maps
        self._density_map = np.ones((numY, numX))
        self._number_map = np.ones((numY, numX))
        self.enumerate_number_map()

        ## Start a dictionary containing the density map objects
        ## Enumerate the dictionary with each of the pharmacophores and objects
        self._pharmacophores = {}
        self.enumerate_density_maps()

        # Start dictionary with query fingerprints
        self._query_fps = {}
        self._selection = False
        self._selection_dict = False
        self._seed = seed
        self._selection_method = None
        self._ref_table = None
        self._coverage = {}

    def pharmacophore(self, pharmacophore):
        """
        return density_map object associated with pharmacophore
        
        Args:
            pharmacophore (str): name of pharmacophore to return
        """
        return self._pharmacophores[pharmacophore]

    ## Coverage functions

    def Coverage(self, bin_loc, ref_loc):
        """
        Initiate the coverage function by providing the necessary information to find data in the database
        This opens the reference and binary files, so may be memory intensive
        """
        self._binary_db = np.memmap(bin_loc, dtype = np.ubyte, mode = 'r')
        with open(ref_loc) as inref:
            self._ref_table, _ = create_ref_table(inref, {})
        self._fp_bytes = int((self._full_fp_length + self._excess_bits) / 8)
        
    def get_compound_fp_dict(self, molname):
        """
        Retrieve the fingerprint for a compound, split it into its constituent pharmacores and convert to map.
        Create a dictionary containing the full fingerprint, pharmacophore fingerprint, 2D fingerprint arrays

        Args:
            molname (str): ID of the compounds / molname to retrieve

        Return: dict containing the full fp, pharmacophore fps and 2D arrays

        """
        try:
            byte_position = int(self._ref_table.loc[molname,'fp_start_position'].min() / 8)
        except:
            logging.error(f"Unable to find molname {molname} in reference table")
            return

        target_fp = get_fp_from_memory_mapped_db(binary_database = self._binary_db, start_position = byte_position, legnth = self._fp_bytes) 
        target_pharmacophore_fps = {}
        target_pharmacophore_fps[molname] = {}
        target_pharmacophore_fps[molname]['pharmacophores'] = {}
        target_pharmacophore_fps[molname]['full_fp'] = target_fp
        pharmacophore_heatmap = np.ones((self._numY,self._numX))

        for pharmacophore in self._pharmacophore_dict.keys():
            start_position = int(self._pharmacophore_dict[pharmacophore]*self._fp_area)
            end_position = int(start_position + self._fp_area)
            pharmacophore_fp = target_fp[start_position:end_position]
            target_pharmacophore_fps[molname]['pharmacophores'][pharmacophore] = {}
            target_pharmacophore_fps[molname]['pharmacophores'][pharmacophore]['fingerprint'] = pharmacophore_fp

            # Translate to a numpy array by creating a 2D array and enumerating it
            target_pharmacophore_fps[molname]['pharmacophores'][pharmacophore]['map'] = np.copy(pharmacophore_heatmap)
            counter = 0
            for y in range(self._numY):
                for x in range(self._numX):
                    try:
                        ## Convert to integer because it gets called as a boolean
                        target_pharmacophore_fps[molname]['pharmacophores'][pharmacophore]['map'][y,x] = int(target_pharmacophore_fps[molname]['pharmacophores'][pharmacophore]['fingerprint'][counter])
                    except:
                        logging.error(f'Error enumerating untrimmed map for pharmacophore {pharmacophore} position {x,y} from density fingerprint position {counter}')
                    counter += 1

        return target_pharmacophore_fps

    def enumerate_selection_fps(self, molnames):
        """
        Populate the selection fingerprint dictionary with the fingerprints of a list of compounds
        This list is used for the purpose of enumerating the fingerprints for a selection
        """
        self._selection_fingerprints = {}
        for molname in molnames:
            try:
                molname_fp_dict = self.get_compound_fp_dict(molname)
                self._selection_fingerprints.update(molname_fp_dict)
            except:
                logging.error(f'Unable to process molecule {molname}, skipping')
                continue

    def enumerate_coverage(self, molnames):
        """
        Creates arrays to assess pharmacophore coverage for a list of molnames
        """
        self.enumerate_selection_fps(molnames = molnames)
        self._total_coverage_arrays = {}

        for pharmacophore in self._pharmacophore_dict.keys():
            self._total_coverage_arrays[pharmacophore] = np.zeros((self._numY, self._numX))

        for molname in self._selection_fingerprints.keys():
            for pharmacophore in self._selection_fingerprints[molname]['pharmacophores']:
                self._total_coverage_arrays[pharmacophore] = self._total_coverage_arrays[pharmacophore] + self._selection_fingerprints[molname]['pharmacophores'][pharmacophore]['map']
                array = np.where(self._total_coverage_arrays[pharmacophore] > 0, 1, np.NaN)
                cmp_array = np.where(self.pharmacophore(pharmacophore)._untrimmed_density_map>0, 1, np.NaN)

                try:
                    pct_coverage = round((np.count_nonzero(~np.isnan(array)) / np.count_nonzero(~np.isnan(cmp_array))) * 100)
                except ZeroDivisionError:
                    pct_coverage = 0

                self._coverage[pharmacophore] = pct_coverage
    def visualise_pharmacophore_coverage(self, pharmacophore, new_style = True):
        """
        Create visualisation of pharmacophore coverage.

        Args:
            pharmacophore(str): Pharmacophore of interest. Use "*" for all

        Return:
            axes object containing visualisation
        """
        if pharmacophore == "*":
            num_pharmacophores = len(self._pharmacophore_dict.keys())

            nrows = math.ceil(num_pharmacophores/2)

            ## Create subplot object
            f, (axes) = plt.subplots(nrows = nrows, ncols = 2, sharey = True, sharex = True)

            ## Each pharmacophore gets its own set of axes
            for pharmacophore, ax in zip(self._pharmacophore_dict.keys(), chain.from_iterable(axes)):
                pharmacophore_name = cp.deepcopy(pharmacophore)

                if new_style:
                    ## New style just shows the coverage, rather than the density of the coverage. Looks better.
                    array = np.where(self._total_coverage_arrays[pharmacophore] > 0, 1, np.NaN)
                else:
                    array = np.where(self._total_coverage_arrays[pharmacophore] == 0, np.NaN, self._total_coverage_arrays[pharmacophore])

                cmp_array = np.where(self.pharmacophore(pharmacophore)._untrimmed_density_map>0, 1, np.NaN)
                
                try:
                    pct_coverage = round((np.count_nonzero(~np.isnan(array)) / np.count_nonzero(~np.isnan(cmp_array))) * 100)
                except ZeroDivisionError:
                    pct_coverage = 0

                self._coverage[pharmacophore] = pct_coverage


                if new_style:
                    ## Set up the background map (overall coverage)
                    one = sns.heatmap(cmp_array, annot = False, fmt = 'g', cmap = 'gist_gray', robust = False, cbar = False, ax = ax, alpha = 0.1, square = True)
                    ## Set up the foreground map (selection coverage)
                    two = sns.heatmap(array, fmt = 'g', cmap = 'Accent', robust = False, cbar = False, annot = False, ax = ax, alpha = 0.8, square = True)
                else:
                    ## Set up the background map (overall coverage)
                    one = sns.heatmap(cmp_array, annot = False, fmt = 'g', cmap = 'twilight', robust = True, cbar = False, ax = ax, square = True)
                    ## Set up the foreground map (selection coverage)
                    two = sns.heatmap(array, cmap = 'summer', robust = False, cbar = False, annot = False, ax = ax, square = True)

                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(f'{pharmacophore_name} - {pct_coverage}% coverage')
                plt.gca().invert_yaxis()

                
            f.tight_layout()
            return f

        else:
            
            array = np.where(self._total_coverage_arrays[pharmacophore] == 0, np.NaN, self._total_coverage_arrays[pharmacophore])
            ax = plt.axes()
            cmp_array = np.where(self.pharmacophore(pharmacophore)._untrimmed_density_map>0, 1,self.pharmacophore(pharmacophore)._untrimmed_density_map)
            cmp_array = np.where(cmp_array==0, np.NaN,cmp_array)
            sns.heatmap(cmp_array, annot = False, fmt = 'g', cmap = 'twilight', robust = True, cbar = False, ax = ax)
            sns.heatmap(array, cmap = 'summer', robust = False, cbar = False, annot = True, ax = ax)
            plt.gca().invert_yaxis()
            return ax

    def visualise_single_compound_coverage(self, molname, pharmacophore):
        """
        Visualise the bbGAP 2D fingerprint for a single compound

        Args:
            molname (str): ID or molname for the molecule
            pharmacophore (str): Pharmacophore to view.
        """
        fp_dict = self.get_compound_fp_dict(molname = molname)
        array = fp_dict[molname]['pharmacophores'][pharmacophore]['map']
        ax = plt.axes()
        sns.heatmap(array, cmap = 'Greens', robust = False, cbar = False, ax = ax)
        return ax

    ## Chemical space functions
      
    def enumerate_number_map(self):
        """
        Enumerate an array which maps where the cell-numbers will be in the y,x numpy array
        """

        ## Starts at 0, which is the 1st position of the list (which is the fingerprint)
        counter = 0
        for y in range(self._numY):
            for x in range(self._numX):
                self._number_map[y,x] = counter
                counter += 1

    def enumerate_density_maps(self):
        """
        Iterate over the pharmacophores in the dictionary
        Take a 'snip' of the full density fingerprint corresponding to the region associated with the
        pharmacophore. This is saved in the pharmacophore dict
        Transform the density fingerprint to a map going from x -> y-
        Enumerate the self._pharmacophores dictionary, keyed by pharmacophore, containing density_map object

        """
        for pharmacophore in self._pharmacophore_dict.keys():
            fp_start_position = int(self._pharmacophore_dict[pharmacophore] * self._fp_area)
            pharmacophore_density_fp = self._fp[fp_start_position:fp_start_position+self._fp_area]
            self._pharmacophores[pharmacophore] = density_map(
                pharmacophore = pharmacophore, 
                pharmacophore_posn = self._pharmacophore_dict[pharmacophore],
                pharmacophore_density_fp = pharmacophore_density_fp, 
                untrimmed_number_map = self._number_map,
                full_fp_length = self._full_fp_length)
         
    def density_heatmaps(self, mask = False, trimmed = True, new_style = True, cmap = 'coolwarm', robust = True, **kwargs):
        """
        Visualise the total occupancy (density) heatmaps across all pharmacophores.
        This will only work if there are 6 pharmacophores being considered!!
        Args:
            mask (bool): whether to show two colours for occupied and unoccupied
            trimmed (bool): whether to show the trimmed density maps or untrimmed
            cmap (str): cmap to use from matplotlib
            robust (bool): arg for matplotlib
        """
        
        if mask == False:
            vmax = None
        else:
            vmax = 1
        ## Create a multi-plot density heatmap for full pharmacophore maps
        
        ## Create a set of 6 plots in a subplot. Laid out with n rows and 2 columns, depending on number of pharmacophores

        num_pharmacophores = len(self._pharmacophores.keys())

        nrows = math.ceil(num_pharmacophores/2)

        ## Create subplot object
        f, (axes) = plt.subplots(nrows = nrows, ncols = 2, sharey = True, sharex = True)
        
        ## Each pharmacophore gets its own set of axes
        for pharmacophore, ax in zip(self._pharmacophores.keys(), chain.from_iterable(axes)):

            if trimmed:
                array = self._pharmacophores[pharmacophore].density_map()
            else:
                array = self._pharmacophores[pharmacophore].untrimmed_density_map()
            if new_style:
                array = np.where(array > 0, array, np.NaN)
                #linewidths = 0#.005
                #cmap = 'PuBuGn'
            else:
                None
                #linewidths = 0

            pharmacophore_name = cp.deepcopy(pharmacophore)
            pharmacophore = sns.heatmap(array,
                                        cmap = cmap,
                                        ax = ax, 
                                        cbar = False,
                                        square = True,
                                        robust = robust,
                                        vmax = vmax,
                                        linecolor = 'gainsboro',
                                        **kwargs)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title(pharmacophore_name)
            plt.gca().invert_yaxis()
        f.tight_layout()
        return f

    ## Clustering and selection functions

    def selection_partition_maps(self, untrimmed = True):
        """
        Visualise the division of the resulting clusters of the compound sets
        Creates a set of 6 plots in subplots.
        Args:
            untrimmed (bool): whether to use the full pharmacophore map or not.
        
        """
        ## Determine the current selection mode.
        if self._selection_method == None:
            logging.error("Please instruct on a selection method. Use class method classic_partitioning() or som_partitioning()")
        else:
            method = self._selection_method

        ## Set the format for the annotation
        fmt = 's'

        ## Determine which pharmacophores were selected from:
        select_pharmacophores = [pharmacophore for pharmacophore, selection in filter(lambda item: item[1] != 0 , self._selection_dict.items())]
        self._select_pharmacophores = select_pharmacophores

        num_pharmacophores = len(self._pharmacophores.keys())

        nrows = math.ceil(num_pharmacophores/2)

        ## Create a set of plots in a subplot. Laid out with n rows and 2 columns, depending on number of pharmacophores.
        f, (axes) = plt.subplots(nrows = nrows, ncols = 2, sharey = True, sharex = True)

        for pharmacophore, ax in zip(self._pharmacophores.keys(), chain.from_iterable(axes)):

            if pharmacophore not in select_pharmacophores:

                if untrimmed:
                    array = self.pharmacophore(pharmacophore)._untrimmed_density_map
                else:
                    array = self.pharmacophore(pharmacophore)._density_map

                array = np.where(array == 0, np.NaN, array)

                pharmacophore_name = cp.deepcopy(pharmacophore)

                pharmacophore = sns.heatmap(array,
                                            cmap="twilight",
                                            ax=ax, 
                                            cbar = False,
                                            square = True,
                                            robust = True,
                                            vmin =0,
                                            vmax = 1,
                                            fmt = fmt)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(pharmacophore_name)
                plt.gca().invert_yaxis()

                continue

            elif method == 'space':
                
                fmt = 's'
                if untrimmed:
                    array = untrim_array(
                        self.pharmacophore(pharmacophore)._classic_partition_map_string, 
                        self.pharmacophore(pharmacophore)._number_map,
                        self.pharmacophore(pharmacophore)._untrimmed_number_map,
                        string_type = True)
                else:
                    array = self.pharmacophore(pharmacophore)._classic_partition_map_string
            
                clusters = np.unique(array)

                ## Randomise clusters so that the colours have more contrast
                clusters = random.sample(sorted(clusters),len(clusters))

                counter = 1

                array_as_numbers = np.zeros((array.shape))

                ## Each cluster has to be assigned a number otherwise the plot won't work
                for cluster in clusters:
                    coords = np.where(array == cluster)
                    array_as_numbers[coords] = counter
                    counter += 1

                # get rid of zero areas
                if untrimmed:
                    array_as_numbers[np.where(self.pharmacophore(pharmacophore)._untrimmed_density_map == 0)] = np.NaN
                else:
                    array_as_numbers[np.where(self.pharmacophore(pharmacophore)._density_map == 0)] = np.NaN
                
                array = array_as_numbers

                pharmacophore_name = cp.deepcopy(pharmacophore)

            else:  
                if untrimmed:
                    number_array = untrim_array(
                        self.pharmacophore(pharmacophore)._SOM_partition_map, 
                        self.pharmacophore(pharmacophore)._number_map,
                        self.pharmacophore(pharmacophore)._untrimmed_number_map,
                        string_type = False)
                    annot_array = untrim_array(
                        self.pharmacophore(pharmacophore)._SOM_partition_map_string, 
                        self.pharmacophore(pharmacophore)._number_map,
                        self.pharmacophore(pharmacophore)._untrimmed_number_map,
                        string_type = True)

                else:
                    number_array = self.pharmacophore(pharmacophore)._SOM_partition_map
                    annot_array = self.pharmacophore(pharmacophore)._SOM_partition_map_string
                array = np.where(density_map == 0, np.NaN, number_array)
                array = np.where(array == 0, np.NaN, array)
                annot = annot_array
                fmt = 's'

                pharmacophore_name = cp.deepcopy(pharmacophore)
            
            pharmacophore = sns.heatmap(array,
                                        cmap="Spectral",
                                        ax=ax, 
                                        cbar = False,
                                        square = True,
                                        robust = True,
                                        vmax = None,
                                        #annot = annot,
                                        fmt = fmt)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title(pharmacophore_name)
            plt.gca().invert_yaxis()
        f.tight_layout()
        return f
                    
    def som_partitioning(self, log = 2, sigma = 1, learning_rate = 0.5, num_iteration = 'len', topology = 'rectangular', random_weights_init = True, neighborhood_function = 'gaussian', activation_distance = 'euclidean', big_x = True, multiplier = 1, seed = False, transform_func = None):
        """
        Perform partitioning using the SOM.
        Refer to train_SOM() function for information about the arguments.
        """

        logging.info("Starting set-driven division of area")

        self._selection_method = 'som'
        self._sigma = sigma
        self._learning_rate = learning_rate
        self._log = log
        self._num_iteration = num_iteration

        ## If a seed is not set, use default

        if not seed:
            seed = self._seed

        ## For pharmacophores that are going to be selected from:
        for pharmacophore in self.GetSelection().keys():

            ## Start dictionary containing the query fingerprints
            self._query_fps[pharmacophore] = {}

            if self.GetSelection()[pharmacophore] == 0:
                continue

            # Initiate and train the SOM
            self.pharmacophore(pharmacophore).som_partitioning(
                selection_size = self.GetSelection()[pharmacophore], 
                log = log, 
                sigma = sigma, 
                learning_rate = learning_rate, 
                num_iteration = num_iteration, 
                topology = topology, 
                random_weights_init = random_weights_init, 
                neighborhood_function = neighborhood_function, 
                activation_distance = activation_distance, 
                big_x = big_x,
                seed = seed,
                transform_func = transform_func)

            # Get the resulting fingerprints
            pharmacophore_fps = self.pharmacophore(pharmacophore)._set_driven_clustering_fps

            # Iterate over partial fingerprints and convert to full fingerprints
            for fp_ID in pharmacophore_fps.keys():
                full_fp = FullFPfromPharmacophoreFP(
                        pharmacophore = pharmacophore,
                        pharmacophore_dict = self._pharmacophore_dict,
                        pharmacophore_fp = pharmacophore_fps[fp_ID],
                        excess_bits = self._excess_bits)
                self._query_fps[pharmacophore][fp_ID] = bitarray(full_fp)

    def classic_partitioning(self, dimension_ratio = 0.6, trimmed = True, method = 'round'):
        """
        Perform classical partitioning of the information.
        Refer to the partition_array() function for details of the arguments.
        """

        logging.info("Starting space-driven division of area")

        self._selection_method = 'space'

        ## For pharmacophores that are going to be selected from:
        for pharmacophore in self.GetSelection().keys():

            self._query_fps[pharmacophore] = {}

            logging.debug(f'Division of {pharmacophore} into {self.GetSelection()[pharmacophore]}')

            if self.GetSelection()[pharmacophore] == 0:
                continue

            # Divide the area
            self.pharmacophore(pharmacophore).classic_partitioning(
                selection_size = self.GetSelection()[pharmacophore],
                dimension_ratio = dimension_ratio, 
                trimmed = trimmed,
                method = method)

            # Get the resulting fingerprints
            pharmacophore_fps = self.pharmacophore(pharmacophore)._space_driven_clustering_fps

            # Iterate over partial fingerprints and convert to full fingerprints
            for fp_ID in pharmacophore_fps.keys():
                full_fp = FullFPfromPharmacophoreFP(
                        pharmacophore = pharmacophore,
                        pharmacophore_dict = self._pharmacophore_dict,
                        pharmacophore_fp = pharmacophore_fps[fp_ID],
                        excess_bits = self._excess_bits)
                self._query_fps[pharmacophore][fp_ID] = bitarray(full_fp)

    def full_coverage(self):
        self._selection_method = 'full_coverage'

    def SetSelection(self, selection, biased_divide = False, p_select = '*', select_mode = 1):
        """
        Set the selection parameters from each of the pharmacophores.

        This can be an integer which will be split evenly between the pharmacophores
        e.g. 96

        Or comma-sep list of pharmacophores with a selection number after a colon

        e.g. 01_A:20,04_D:20,07_-:1,08_+:0,09_R:20,10_H:5

        Args:

            selection(int or str): see above
            p_select (str): pharmacophore to select from
            select_mode (int): how to distribute selection across pharmacophores. 
                options:0 - proportionally based on total area.
                        1 - even
                        2 - proportionally on log scale
                        3 - proportionally
                        note - only 1 should be used. Others for testing.
                        Doing elsewise biases greatly towards hydrophobic
        """

        self._selection = selection
        self._p_select = p_select
        self._select_mode = select_mode

        # Start a dictionary which contains how many compounds to select from each pharmacophore
        self._selection_dict = {}
        # Enumerate the dictionary
        self.determine_selections()

    def determine_selections(self):
        """
        Assign the number of compounds to be selected from each pharmacophore
        """

        ## Start dictionary containing pharmacophores, and set all selections to 0
        for pharmacophore in self._pharmacophore_dict.keys():
            self._selection_dict[pharmacophore] = 0

        ## Try to convert the selection parameter to an integer.
        try:
            self._selection = int(self._selection)
        except:
            pass

        # If an integer is provided, proceed with current (not old) allocation method.
        if type(self._selection) == int:
            
            # If selecting from all the pharmacophores
            if self._p_select == '*':
                self._selected_pharmacophores = list(self._pharmacophore_dict.keys())

            ## Else parse the list of pharmacophores
            else:
                self._selected_pharmacophores = self._p_select.split(',')
                for pharmacophore in self._selected_pharmacophores:
                   if pharmacophore not in self._pharmacophore_dict.keys():
                        logging.error(f'Pharmacophore {pharmacophore} not in available pharmacophores, quitting')
                        sys.exit(2)

            ## Check that all the pharmacophores to select from are able to be selected from the set
            for pharmacophore in self._selected_pharmacophores:
                ## If there is no density in this pharmacophore map
                if len(self.pharmacophore(pharmacophore)._density_map) == 0:
                    logging.warning(f"There are no compounds containing pharmacphore {pharmacophore}. None will be selected")
                    self._selected_pharmacophores.remove(pharmacophore)

            #-----These methods differ in how they allocate the selections from each pharmacophore-----

            ## If desired number of compounds are indicated from each pharmacophore from entering 01_A:20, etc
            if ":" in self._p_select:
                for pharm_selection in self._selected_pharmacophores:
                    pharmacophore,selection = pharm_selection.split(':')
                    if pharmacophore not in self._pharmacophore_dict.keys():
                        logging.error(f'Pharmacophore {pharmacophore} not in available pharmacophores, quitting')
                        sys.exit(2)
                    self.GetSelection()[pharmacophore] = int(selection)

            # If in the biased allocation mode 
            # allocations are made based on the number of compounds within each pharmacophore.
            elif self._select_mode >=2:

                self.BiasedAllocation()

            ## If in even allocation mode 
            ## Same number of selections from each pharmacophore):
            elif self._select_mode == 1:

                # Count number of pharmacophores to select from
                num_pharmacophores = len(self._selected_pharmacophores)

                # Divide total selection by number of pharmacophores.
                # Round the number up for each selection
                selection_per_pharmacophore = math.ceil(self._selection/num_pharmacophores)

                # Iterate over the selected pharmacophores and enumerate the selection dictionary
                for pharmacophore in self._selected_pharmacophores:
                    self.GetSelection()[pharmacophore] = selection_per_pharmacophore

            ## If performing allocation based on the chemical space coverage of each pharmacophore

            elif self._select_mode == 0:

                self.SpaceBasedAllocation()

            else:
                logging.error("Unable to calculate the selections from each pharmacophore due to propblems parsing the selection paramaters. Please check input and try again")
                exit(2)

        ## THIS IS FOR BACKWARD COMPATIBILITY
        else:
            try:

                if "-" in self._selection:
                    ## Biased selection based on the set

                    ## If -- is used, do a log of the totals, which makes a more even selection
                    if "--" in self._selection:
                        ## Capture log condition
                        self._select_mode = 2
                        selections = self._selection.split("--")

                    else:
                        self._select_mode = 3
                        selections = self._selection.split('-')

                    ## The total desired selection size
                    self._selection = int(selections[0])

                    ## If selection is from all pharmacophores, take a copy of the pharmacophores
                    if selections[1] == "*":
                        self._selected_pharmacophores = list(self._pharmacophore_dict.keys())
                    
                    ## Otherwise determine which pharmacophores are to be selected from
                    else:
                        self._selected_pharmacophores = selections[1].split(',')

                    self.BiasedAllocation()
            except:
                logging.error(f'Unable to parse selection shorthand input {self._selection}\n'+
                              f'Format "pharmacophoreID:selection" (comma sep) to choose selections from pfs\n'+
                              f'e.g. 01_A:20,04_D:20,07_-:1,08_+:0,09_R:20,10_H:5\n'+
                              f'OR an integer e.g. 96 to attempt to split evenly between pharmacophores\n'+
                              f'OR an integer-pharmacophores. to split based on set. Use "*" for all pharmacophores\n'+
                              f'e.g. 96-01_A,04_D or 96-*. Use double -- to log values and given more even')
                traceback.print_exc()
                sys.exit(2)

    def SpaceBasedAllocation(self):
        """
        Updates self._selection_dict by performing an based on chemical space coverage of the pharmacophores
        This depends on the number of "cells" covered by each pharmacophore and allocates the selections to the pharmacophores
        dependant on that.

        """
        selection_size = self._selection

        ## Keep a dictionary containing the areas of the chemical space coverage
        self._pharmacophore_area_dict = {}
        pharmacophore_size_list = []

        ## Keep track of the total chemical space covered by the set
        self._total_chemical_space_area = 0

        ## Keep track of the number of allocations that have been made.
        ## This is used to determine how many remain after 
        total_allocations = 0
        

        ## Iterate over the pharmacophores to select from and determine their chemical space coverage
        for pharmacophore in self._selected_pharmacophores:
            self._pharmacophore_area_dict[pharmacophore] = {}

            ## count non-zero cells in the pharmacophore to determine chemical space coverage
            area = np.count_nonzero(self._pharmacophores[pharmacophore]._density_map)
            self._pharmacophore_area_dict[pharmacophore]['area'] = area
            self._total_chemical_space_area += area


        ## Calculate the proportion of the total chemical space coverage for each pharmacophore
        ## Determine how many selections are allocated to each pharmacophore
        for pharmacophore in self._pharmacophore_area_dict.keys():


            proportion = self._pharmacophore_area_dict[pharmacophore]['area'] / self._total_chemical_space_area
            self._pharmacophore_area_dict[pharmacophore]['proportion'] = proportion
            allocation = math.floor(proportion*selection_size)
            self._pharmacophore_area_dict[pharmacophore]['allocation'] = allocation

            if self._pharmacophore_area_dict[pharmacophore]['area'] != 0:
                pharmacophore_size_list.append([pharmacophore, self._pharmacophore_area_dict[pharmacophore]['area'], allocation])

            total_allocations += allocation


        ## Determine how many leftover allocations there are            
        remaining_allocations = selection_size - total_allocations

        ## REMOVED Sort the list of pharmacophores, such that those with the lowest number of allocated compounds are allocated more
        #pharmacophore_size_list.sort(key = lambda x: (x[2], -x[1]), reverse = False)
        pharmacophore_size_list.sort(key = lambda x: x[1], reverse = True)

        ## Allocate the remaining selections to the largest chemical spaces
        while remaining_allocations > 0:
            for pharmacophore, area, allocation in pharmacophore_size_list:
                if remaining_allocations == 0:
                    break
                self._pharmacophore_area_dict[pharmacophore]['allocation'] += 1
                remaining_allocations -= 1

        ## Update the selection dictionary
        for pharmacophore in self._pharmacophore_area_dict.keys():
            self._selection_dict[pharmacophore] = self._pharmacophore_area_dict[pharmacophore]['allocation']

    def BiasedAllocation(self):
        """
        Updates self._selection_dict to perform a bias selection. 
        This depends on how many contain the pharmcophores across the set.
        """

        if self._select_mode == 2:
            log_condition = True
        else:
            log_condition = False

        selection_number = self._selection
        selected_pharmacophores = self._selected_pharmacophores

        ## Track total density
        total_density = 0
        pharmacophore_selection_dict = {}

        ## This list will be used if the selection isn't perfect to remove and add from selections
        pharmacophore_selection_list = []

        ## Iterate over desired pharmacophores and gather the sum of their densities
        for pharmacophore in selected_pharmacophores:

            ## Get total density of the pharmacophore
            pharmacophore_total_density = np.array(self._pharmacophores[pharmacophore]._density_fp).astype(int).sum()

            if log_condition:
                if pharmacophore_total_density == 0:
                    pass
                else:
                    ## Take a log of 2, which evens out the densities slightly.
                    pharmacophore_total_density = math.log(pharmacophore_total_density,2)

            pharmacophore_selection_dict[pharmacophore] = {}

            pharmacophore_selection_dict[pharmacophore]['total'] = pharmacophore_total_density

            total_density += pharmacophore_selection_dict[pharmacophore]['total']

            logging.debug(f"total density for pharmacophore {pharmacophore}: {pharmacophore_selection_dict[pharmacophore]['total']}")

        ## Iterate again now knowing the totals to determine the selections for each pharmacophore

        for pharmacophore in selected_pharmacophores:

            ## Calculate the proportion of the total density for the pharmacophore
            pharmacophore_selection_dict[pharmacophore]['proportion'] = pharmacophore_selection_dict[pharmacophore]['total']/total_density

            ## Calculate the selection based on the proportion. Round to nearest integer
            pharmacophore_selection_dict[pharmacophore]['selection'] = round(pharmacophore_selection_dict[pharmacophore]['proportion'] * selection_number)

            ## Track the pharmacophore and the selection size (non-rounded)
            pharmacophore_selection_list.append([pharmacophore,(pharmacophore_selection_dict[pharmacophore]['proportion'] * selection_number)])
            logging.debug(f"initial selection for pharmacophore {pharmacophore} = {pharmacophore_selection_dict[pharmacophore]['selection']}")


        ## Find if the number of selections satisfies the selection size:

        total_selections = np.array([int(pharmacophore_selection_dict[i]["selection"]) for i in pharmacophore_selection_dict.keys()]).sum()
                  
        logging.debug(f'initial selection: {total_selections}')

        excess_selection = total_selections - selection_number
        logging.debug(f'excess selection = {excess_selection}')
                  
        ## If no compounds were selected in first round (because rounding took them all to zero)
        ## Iterate from most dense and add one to each of the most dense groups
        if total_selections == 0:
            for pharmacophore in sorted(pharmacophore_selection_list, key = lambda x: x[1], reverse = False)[excess_selection:]:
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] += -1 * (excess_selection / abs(excess_selection))

        ## If there are too few selected compounds, iterate from the pharmacophores with the least selected and give them an extra one

        elif excess_selection < 0:
                  
            for pharmacophore in sorted(pharmacophore_selection_list, key = lambda x: x[1], reverse = True)[excess_selection:]:
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] += -1 * (excess_selection / abs(excess_selection))

        ## If there are too many selected, and all pharmacophores have been selected from and some more than once, 
        ## iterate from the pharmacophores with the most selected and remove one
        elif excess_selection > 0 and total_selections > len(selected_pharmacophores):
                  
            for pharmacophore in sorted(pharmacophore_selection_list, key = lambda x: x[1], reverse = True)[:excess_selection]:
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] += -1 * (excess_selection / abs(excess_selection))
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] = int(pharmacophore_selection_dict[pharmacophore[0]]['selection'])
                  
        ## If there are the too many selected, but not all the pharmacophores have been selected from, 
        ## remove from the least dense
                  
        elif excess_selection > 0 and total_selections <= len(selected_pharmacophores):
                  
            for pharmacophore in sorted(pharmacophore_selection_list, key = lambda x: x[1], reverse = False):
                if pharmacophore_selection_dict[pharmacophore[0]]['selection'] <= 0:
                  continue
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] += -1
                pharmacophore_selection_dict[pharmacophore[0]]['selection'] = int(pharmacophore_selection_dict[pharmacophore[0]]['selection'])
                excess_selection -= 1
                if excess_selection == 0:
                  break

        ## Iterate over pharmacophores and set the selections

        for pharmacophore in pharmacophore_selection_dict.keys():
            logging.debug(f'final selection for pharmacophore {pharmacophore} = {pharmacophore_selection_dict[pharmacophore]["selection"]}')

            ## Update selection dictionary
            self.GetSelection()[pharmacophore] = pharmacophore_selection_dict[pharmacophore]["selection"]

        logging.debug(f'final selection: {np.array([int(pharmacophore_selection_dict[i]["selection"]) for i in pharmacophore_selection_dict.keys()]).sum()}')

    def GetSelection(self):
        if self._selection == False or self._selection_dict == False:
            logging.error(f'Selection not specified. Use SetSelection() method')
        else:
            return self._selection_dict

    def GetQueryFps(self):
        if not self._query_fps:
            logging.error(f'Clustering not performed yet. Use .som_partitioning() or .classic_partitioning() methods')
        else:
            return self._query_fps

    #def seed(self, seed):
        """
        Set random seed. Not used.
        """
    #    self._seed = seed


    def GetCompoundFullFP(self, molname):
        """
        Returns a compound's fingerprint. Requires the Coverage method to have been applied to load in the binary file.

        Args:
            molname (str): the ID of the molecule
        Returns:
            BitArray object containing compound's full fP
        """

        ## Check if reference table has been set up
        try:
            ## Get the byte position
            byte_position = int(self._ref_table.loc[molname,'fp_start_position'].min() / 8)
            ## Get the fp
            target_fp = get_fp_from_memory_mapped_db(binary_database = self._binary_db, start_position = byte_position, legnth = self._fp_bytes)
            return target_fp
        except:
            traceback.print_exc()
            logging.error(f'Correct database has not been loaded into the class or compound not found. Use Coverage() method.')

    def GetRefTable(self):
        """
        Get the reference table
        Returns:
            pd.DataFrame with the compounds and associated data.
        """
        if not isinstance(self._ref_table, pd.DataFrame):
            logging.error('Ref table has not yet been created. Ensure Coverage() method has been initiated')
            raise RuntimeError('Reference table has not been added.')
        else:
            return self._ref_table


    def get_rdkit_fp(self, molname, bleed = False, pharmacophore = '*'):
        """
        Returns an RDkit fingerprint from a molname

        Args:
            molname: the ID of the molecul

        Returns:
            ExplicitBitVect
        """
        if pharmacophore == '*':
            fp = self.GetCompoundFullFP(molname)
        else:
            fp = self.get_single_pharmacophore_fp(molname, pharmacophore)

        if bleed == False:
            return bitarray_to_rdkit_fp(fp)

        else:
            bbSelect_str_fp = fp.to01()
            bbSelect_pharmacophore_fp = {}
            for pharmacophore in self._pharmacophore_dict.keys():
                bbSelect_pharmacophore_fp[pharmacophore] = {}
                start_number = self._fp_area * self._pharmacophore_dict[pharmacophore]
                end_number = start_number + self._fp_area
                
                bbSelect_pharmacophore_fp[pharmacophore]['fp'] = bbSelect_str_fp[start_number:end_number]
                
                bbSelect_pharmacophore_fp[pharmacophore]['fp_array'] = np.zeros((self._numY, self._numX))
                
                bit_position = 0
                for y in range(self._numY):
                    for x in range(self._numX):
                        bit = bbSelect_pharmacophore_fp[pharmacophore]['fp'][bit_position]
                        bbSelect_pharmacophore_fp[pharmacophore]['fp_array'][y,x] = bit
                        bit_position +=1
                
                bbSelect_pharmacophore_fp[pharmacophore]['fp_bleed_array'] = bleed_array(bbSelect_pharmacophore_fp[pharmacophore]['fp_array'])
            full_bleed_array = []
            for pharmacophore in bbSelect_pharmacophore_fp.keys():
                for row in bbSelect_pharmacophore_fp[pharmacophore]['fp_bleed_array']:
                    full_bleed_array.extend(list(row))
            full_bleed_list = [str(int(i)) for i in full_bleed_array]
            full_bleed_fp = "".join(full_bleed_list)+'0'*self._excess_bits
            return DataStructs.cDataStructs.CreateFromBitString(full_bleed_fp)

    def get_single_pharmacophore_fp(self, molname, pharmacophore):
        """
        Get a molecule's fingerprint for a single pharmacophore

        Args:
            molname (str): ID or molname for molecule
            pharmacophore (str): which pharmacophore to use

        """

        compound_fp = self.GetCompoundFullFP(molname)

        dummy_fingerprint = self.create_dummy_fingerprint(pharmacophore)

        # Get the product of the two fingerprints
        ## This means that only positions bpth containing 1s are kept
        query_fp_bitarray = dummy_fingerprint & compound_fp

        # Convert to RDKit fingerprint
        return query_fp_bitarray

    def create_dummy_fingerprint(self, pharmacophore):
        """
        Create an dummy fingerprint where only the bits related to the pharmacophore are turned on.

        Args:
            pharmacophore (str): Pharmacophore feature to use
        """

        pharmacophores = pharmacophore.split(',')

        # Create the dummy fingerprint that will be used

        feature_fp_length = self._fp_area

        # Create an empty version of the fingerprint
        empty_pharmacophore_fp = [0]*self._fp_area

        # Start with a list of [0]'s equal to the number of pharmacophores
        empty_fp = [0]*len(self._pharmacophore_dict.keys())

        # Enumerate a list of lists containing an empty pharmacophore fp at each pharmacophore position
        for i, position in enumerate(empty_fp):
            empty_fp[i] = empty_pharmacophore_fp

        for pharmacophore_name in pharmacophores:
            if pharmacophore_name not in self._pharmacophore_dict.keys():
                logging.error(f'Pharmacophore {pharmacophore_name} does not exist. Check spelling')
                raise ValueError(f'Pharmacophore {pharmacophore_name} does not exist. Check spelling')
                
            # Insert 1s into the pharmacophore region of interest
            empty_fp[self._pharmacophore_dict[pharmacophore_name]] = [1]*self._fp_area

        # merge lists of lists into a single list
        full_fp = list(chain(*empty_fp))

        # Add excess bits
        full_fp.extend([0]*self._excess_bits)

        # Convert to bitarray object

        return bitarray(full_fp)

    def calculate_similarity_to_molecule(self, query_molname, return_mols = False, metric = "Tanimoto", fuzzy = False, prefix = "bbSelect_similarity", a = 0, b = 1, pharmacophore = '*'):
        """
        Calculates the similarity of a compound in the bbGAP database to the other compounds in that database.

        Args:
            query_molname (str): compound to query
            return_mols (bool): whether to add rdkit.Molecule objects to the returns pd.DataFrame
            metric (str): which metric to caclulate similarity. Between "Tanimoto" and "Tversky"
            fuzzy (bool): whether to use a fuzzy similarity in 2D to match compounds which place a pharmacophore adjacent
            prefix (str): the prefix to use for the returned similarity in the pd.DataFrame
            a (float): the value for a in the Tversky similarity
            b (flot): the value for b in the Tversky similarity
            pharmacophore (str): which pharmacophore to use. use * for all

        """

        ## Get the query FP 
        query_fp = self.get_rdkit_fp(query_molname, fuzzy, pharmacophore = pharmacophore)

        ## Track matches
        matches = []

        ## Iterate over the ref table and compare the fingerprints of each of the molecules (SLOW COULD BE PARALLELISED)
        for molname in self.GetRefTable().index:

            ## Get the Rdkit fp for the molecule
            target_fp = self.get_rdkit_fp(molname, pharmacophore = pharmacophore)
            arr1 = np.zeros((0,), dtype=np.int8)
            arr2 = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(target_fp,arr1)
            DataStructs.ConvertToNumpyArray(query_fp,arr2)
            if metric == 'Tversky': 
                similarity = DataStructs.TverskySimilarity(query_fp, target_fp, a, b)
            elif metric == 'Tanimoto':
                similarity = DataStructs.FingerprintSimilarity(query_fp,target_fp, metric=DataStructs.TanimotoSimilarity)
            else:
                raise RuntimeError(f'Metric {metric} not supported')
            smiles = self._ref_table.loc[molname]['smiles']

            if isinstance(smiles, list) or not isinstance(smiles, str) :
                logging.error(f'smiles for {molname}, {smiles} is a {type(smiles)}, taking the first smiles {smiles[0]}')
                smiles = smiles[0]

            matches.append([molname, smiles, similarity])

        ## Sort the matches by similarity
        matches.sort(key = lambda x: x[2], reverse = True)

        matches_table = pd.DataFrame(matches, columns = ['ID', 'smiles', f'{prefix}_{query_molname}'])

        if return_mols:

            PandasTools.AddMoleculeColumnToFrame(matches_table,'smiles','Molecule',includeFingerprints=True)

        return matches_table

## These functions are for the constructions of density maps

def FullFPfromPharmacophoreFP(pharmacophore, pharmacophore_dict, pharmacophore_fp,excess_bits):
    """
    Converts a single pharmacophore fingerprint to a full legnth fingerprint
    
    Args:
        pharmacophore (str): ID of pharmacophore
        pharmacophore_dict (dict): dictionary keyed by pharmacophores and containing the positions of the pharmacophores within the full fingerprint
        pharmacophore_fp (np.Array): partial query fingerprint (for pharmacophore)
        excess_bits (int): the number of excess bits needed to convert bits to bytes
    
    Returns:
        full fingerprint as a list
    """

    # Create an empty version of the fingerprint
    empty_pharmacophore_fp = [0]*len(pharmacophore_fp)

    # Start with a list of [0]'s equal to the number of pharmacophores
    empty_fp = [0]*len(pharmacophore_dict.keys())

    # Enumerate a list of lists containing an empty pharmacophore fp at each pharmacophore position
    for i, position in enumerate(empty_fp):
        empty_fp[i] = empty_pharmacophore_fp

    # Inject partial pharmacophore fingerprint into the list of lists:
    empty_fp[pharmacophore_dict[pharmacophore]] = pharmacophore_fp

    # merge lists of lists into a single list
    full_fp = list(chain(*empty_fp))

    # Add excess bits
    full_fp.extend([0]*excess_bits)

    return full_fp

def TrimMap(density_map,number_map):
    """
    Trims a density map to keep only the range of x and y which contains density.
    This removes areas of the map, if they exist, where no monomers  place functionality in.

    Args:
        density_map (np.Array): a numpy array containing the density of pharmacophores in x and y positions
        number_map (np.Array): the related numpy array containing the cell numbers of each position of the aray
        return: a list containing the trimmed density map and the corresponding trimmed number map.
    """
    
    # Capture the shape of the maps
    y_len, x_len = density_map.shape

    # Begin with the smallest possible value of max x & y
    #                 largest possible value of min x & y
    max_x = 0
    min_x = x_len
    max_y = 0
    min_y = y_len

    # Scanning the x and y positions from smallest to largest
    # The idea is that the lowest position on x and y that contains density is captured, giving minimum x and y
    
    for y in range(y_len):
        for x in range(x_len):
            
            # If the density is 0, ignore as this is to be trimmed off if possible
            if density_map[y,x] == 0:
                pass
            else:
                
                # If the value of x is smaller than a previously found minimum x value, keep it
                if x < min_x:
                    min_x = x
                # If the value of y is smaller than a previously found minimum y value, keep it
                if y < min_y:
                    min_y = y
                    
    # Scanning the x and y positions from largest to smallest
    # The idea is that the highest position on x and y that contains density is captured, giving maximum x and y

    for y in range(y_len)[::-1]:
        for x in range(x_len)[::-1]:
            
            # If the density is 0, ignore as this is to be trimmed off if possible
            if density_map[y,x] == 0:
                pass
            else:
                
                # If the value of x is larger than a previously found maximum x value, keep it
                if x > max_x:
                    max_x = x + 1
                # If the value of y is larger than a previously found maximum y value, keep it
                if y > max_y:
                    max_y = y + 1
                    
    # They are stored in a weird format, so convert to integers as these are counts
    density_map = density_map.astype(int)

    # Return the density map and number mapped, trimmed appropriately
    return (density_map[min_y:max_y,min_x:max_x], number_map[min_y:max_y,min_x:max_x])
    
def discretise_data(array,log=None,random=True,multiplier = 1, seed = 3, transform_func = None):
    """
    Convert a 2D array with cells containing density to discrete values with radomly generated points within each cell.
    Can also be transformed on a log scale if desired.
    There are some functions here that are included for backward compatibility.

        array (np.Array): 2D numpy array
        log (str, int or None): if "sqrt": do squareroot, if int: base to log.
        multiplier (float): number to multiple points by. This gives the SOM more training data (not needed!)
        seed : random seed
        transform_func (function): function to transform by
    """

    ## Look at which prime number to use
    ## Random doesn't work here.
    np.random.seed(seed)

    # Track random points
    random_plot = []
    
    # Iterate over the cells
    for y,line in enumerate(array):
        for x, value in enumerate(line):
            # Ignore cell if there is no density
            if value == 0:
                continue
                
            # If there is a log transform to do
            if log == None:
                pass

            elif transform_func != None:
                value = transform_func(value)

            elif isinstance(log, int):
                # Add one just incase there is a density of '1' which would give zero
                value = value + 1
                # Take a log of base 'log'
                value = math.log(value,log)
                ## Round up
                value = math.ceil(value)

            elif log == 'sqrt':
                value = math.sqrt(value)
                value = math.ceil(value)

            else:
                logging.error(f'log value of {log} not recognised. use an integer or "sqrt", otherwise no transform will be applied')

            if multiplier != 1:
                # Multiply if required
                value = value * float(multiplier)
                value = math.ceil(value)
                
            # Generate random array of shape, value, 2(for x and y)
            random_xy = np.random.rand(value,2).T

            ## Adjust for x and y
            random_xy[0] = random_xy[0] + x
            random_xy[1] = random_xy[1] + y

            # Un-transpose to get back into value,2 shape
            random_xy = random_xy.T

            # Add to total plot
            random_plot.extend(random_xy)
            
    # Turn into pandas dataframe
    random_plot = pd.DataFrame(random_plot)
    
    return random_plot

## These functions are for visualisations of the SOMs

def train_SOM(data, x, y, sigma, learning_rate, num_iteration,topology = 'rectangular',random_weights_init = False,neighborhood_function = 'gaussian',activation_distance = 'euclidean', seed = 3):
    """
    Train a SOM on discrete data
    Arguments reflect those in the minisom package.
    
    Args:
        x (int): x dimension of the SOM.

        y (int): y dimension of the SOM.
            
        sigma (float) optional (default=1.0): Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
                                                (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
        learning_rate (float): initial learning rate
                              (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)

        num_iteration (int): Maximum number of iterations (one iteration per sample).
            
        topology (str), optional (default='rectangular'):Topology of the map. 
            Possible values: 'rectangular', 'hexagonal'
            
        neighborhood_function (str), optional (default='gaussian'): Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
            
        activation_distance (str) optional (default='euclidean'): Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

        random_weights_init (bool): whether to start with random weights based on the scatter data. Recommended.

    Returns: Trained MiniSom object
    
    """
    
    # Initiate SOM
    som = MiniSom(x=x, y=y, input_len = data.shape[1], 
                  sigma = sigma, 
                  learning_rate = learning_rate, 
                  random_seed = seed, 
                  topology = topology,
                  neighborhood_function = neighborhood_function,
                  activation_distance = activation_distance)

    # Initiate with random weights based on random selection of points
    if random_weights_init:
        som.random_weights_init(data)

    ## Initiate with weights to span the first two principal components. Removes any variation due to randomness.
    ## Deterministic
    else:
        som.pca_weights_init(data)   

    # Train the SOM
    if type(num_iteration) == str:
        if num_iteration == 'len':
            num_iteration = data.shape[0]
        else:
            len_multiple = int(num_iteration.split('*')[1])
            num_iteration = data.shape[0] * len_multiple

    som.train_random(data = data, num_iteration = num_iteration)

    return som

def SOM_partitions(som,data):
    """
    Associate each cell with its closest neuron

    Args:
        som (minisom): Trained MiniSom object
        data (np.Array): The data in array format (transformed from pd.DataFrame using pd.DataFrame.values)
    Returns:
        pd.DataFrame containing the x,y coordiantes and related neuron for each of the data points
    """
    
    ## Extract which neuron each data point is associated with
    SOM_dictionary = som.win_map(data)
    
    ## Extract the dictionary into a pandas-friendly list
    SOM_partition_list = []
    
    # Iterate over the neurons and extract the features that are associated with it
    for neuron in SOM_dictionary.keys():

        for (x,y) in SOM_dictionary[neuron]:
            SOM_partition_list.append([x,y,neuron])

    ## Import into dataframe
    SOM_partition_dataframe = pd.DataFrame(SOM_partition_list, columns = ['x','y','neuron'])
    
    return SOM_partition_dataframe

def getNeuronCoordinates(som,x,y):
    """
    Find the coordinates of the neurons in the SOM
    
    Args:
        som (minisom): Trained MiniSom object
    Returns:
        pd.DataFrame containing the x,y coordinates for each of the neurons
    """
    neurons_coordinates = som.get_weights()
    neurons = {}
    counter = 1
    for x_c, i in enumerate(neurons_coordinates):
        for y_c, j in enumerate(i):
            #neurons[f'({(x-1)-x_c},{(y-1)-y_c})'] = j
            neurons[f'({x_c},{y_c})'] = j
            counter +=1

    neuron_map = pd.DataFrame.from_dict(neurons, orient = 'index', columns = ['x','y'])
    
    return neuron_map

def visualise_som_scatter(SOM_partition_dataframe, neuron_map, random_order = True):
    """
    Create a visualisation of the clustered scatterplot overlaid with the positions of the SOM neurons

    Args:
    
        SOM_partition_dataframe (pd.DataFrame): pd.DataFrame containing each data point with associated neuron
        neuron_map (pd.DataFrame): pd.DataFrame containing the x,y coordinates for each of the neurons
    """
    ## Take the unique neurons
    unique_neurons = sorted(SOM_partition_dataframe['neuron'].unique())
    
    ## Count the unique neurons
    num_neurons = len(unique_neurons)
    
    ## Return the unique neurons into a random order
    if random_order:
        unique_neurons = random.sample(sorted(unique_neurons),num_neurons)
                             
    z = range(1,num_neurons)
    #Spectral = plt.get_cmap('Spectral')
    Spectral = plt.get_cmap('rainbow')
    cNorm = colors.Normalize(vmin = 0, vmax = num_neurons)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = Spectral)


    neuron_values = [i for i in unique_neurons]

    x = SOM_partition_dataframe['x']
    y = SOM_partition_dataframe['y']
    
    max_x = int(math.ceil(SOM_partition_dataframe['x'].max()))
    x_ticks = [i for i in range(max_x)]
    max_y = int(math.ceil(SOM_partition_dataframe['y'].max()))
    y_ticks = [i for i in range(max_y)]

    for i in range(len(neuron_values)):
        indx = SOM_partition_dataframe['neuron'] == neuron_values[i]
        plt.scatter(x[indx], y[indx], s = 500, alpha = 0.9, color=scalarMap.to_rgba(i), label = neuron_values[i])


    plt.scatter(neuron_map['x'],neuron_map['y'], marker = '+', s = 200, c = '#000000')
    plt.grid(b=True, which='both', color='#D2D2D2', linestyle='-', alpha = 1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

def clusterMapFromSOM(density_map, SOM_partition_dataframe,random_sort = False):
    """
    create the map of partitions using the SOM and the density map
    Args:
        density_map (np.Array) : 2D array contatining the overall occupancy for each pharmacophore
        SOM_partition_dataframe (pd.DataFrame): Contains the SOM clusters

    """
    ## Create an empty numpy array to store the clusters from SOM
    SOM_partition_map = np.zeros(density_map.shape)
    SOM_partition_map_string = np.zeros(density_map.shape, dtype='a10')

    # Get list of neurons, sorted from 0,0
    neuron_list = sorted(SOM_partition_dataframe['neuron'].unique())
    
    # Randomise the list order. This gives a greater contrast when visualising
    if random_sort:
        neuron_list = random.sample(neuron_list, len(neuron_list))

    # Start dictionary of cluster number of neurons
    neuron_dict = {}

    # Enumerate dictionary with the neuron lists 
    for i, neuron in enumerate(neuron_list):
        neuron_dict[str(neuron)] = i+1

    # Iterate over each of the cells in the new array to fill them.

    # For y-coordinate, which is also a self-contained list
    for y, line in enumerate(SOM_partition_map):

        # For x-coordinate, which contain the values
        for x, value in enumerate(line):
            
            # Find points which lie within the cell
            cell_matches = SOM_partition_dataframe[SOM_partition_dataframe['x'].between(x,x+1) & SOM_partition_dataframe['y'].between(y,y+1)]
            
            # Ignore cells which do not contain any points
            if cell_matches.empty:
                continue
                
            # Find the closest SOM neuron, 
            # This is the neuron which most of the scatter points associate with
            mode_neuron = cell_matches['neuron'].mode().values[0]
            
            # Assign a cluster number for the neuron and add it to the cluster map
            mode_neuron_name = neuron_dict[str(mode_neuron)]
            SOM_partition_map[y,x] = mode_neuron_name
            SOM_partition_map_string[y,x] = str(mode_neuron)

            cluster_dict = {v:k for k,v in neuron_dict.items()}

    SOM_partition_map_string = SOM_partition_map_string.astype('str')

    return SOM_partition_map, SOM_partition_map_string, cluster_dict

def visualise_SOM_partitions(SOM_partition_map, SOM_partition_map_string, density_map, annot_density, neuron_coordinates, show_neuron_coordinates = False, cmap = 'Spectral', title = False, ax_alias = False):
    """
    Visualise the partitioning performed by the SOM
    """
    #SOM_partition_map = np.where(SOM_partition_map == 0, np.NaN, SOM_partition_map)
    SOM_partition_map = np.where(density_map == 0, np.NaN, SOM_partition_map)



    if annot_density:
        annot = density_map
        fmt = 'g'
    else:
        annot = SOM_partition_map_string
        fmt = 's'

    if ax_alias:
        ax_name = {}
        ax_name[ax_alias] = plt.axes()

    else:
        ax_alias_dict = {}
        ax_alias_dict['ax'] = plt.axes()

    if ax_alias:
        sns.heatmap(SOM_partition_map, cmap = cmap, cbar = False, square = True, robust = True, annot = annot, fmt = fmt, ax = ax_name[ax_alias])
        if show_neuron_coordinates:
            ax_name[ax_alias].scatter(neuron_coordinates['x'],neuron_coordinates['y'], marker = '+', s = 200, c = '#000000')

    else:
        sns.heatmap(SOM_partition_map, cmap = cmap, cbar = False, square = True, robust = True, annot = annot, fmt = fmt, ax = ax_alias_dict['ax'])

        if show_neuron_coordinates:
            ax_alias_dict['ax'].scatter(neuron_coordinates['x'],neuron_coordinates['y'], marker = '+', s = 200, c = '#000000')

    if title:
        plt.title(title)

    plt.gca().invert_yaxis()

    if ax_alias:
        return ax_name[ax_alias]
    else:
        return ax_alias_dict['ax']

## These functions are for the even division of maps

def find_available_dimensions(n):
    """
    Find the pairs of integers that can be multiplied to make n
    Args:
        n (int): number to find integer divisions
    Returns:
        dimension_list: list containing the pairs of integers which can be multiplied to make n.
    """
    dimension_list = []
    # Iterate over all the integers up to n (excluding 0)
    for i in range(math.ceil(n)):
        # Check if n is divisible by integer, if so, store it
        if n%(i+1) == 0:
            dimension_list.append([int(n/(i+1)),(i+1)])
                
    return dimension_list

def divide_array(array, x_edge, y_edge,fp):
    """
    Divide an array given the sizes of the x and y edges of the sub-areas.
    Return a dictionary which is keyed by the cluster 'ID' and contianing the fingerprint and subarea
    Args:
        array (np.Array): numpy array to divide
        x_edge (float): length of x-axis of sub-area
        y_edge (float): length of y-axis of sub-area
    """
    
    subarea_dict = {}
    chunk_list = []
    counter = 0

    numY,numX = np.shape(array)
    # take steps along the Y axis with the size of the desired edge
    for j in range(numY)[0:numY:y_edge]:
        # take steps along the X axis with the size of the desired edge
        for i in range(numX)[0:numX:x_edge]:
            # take the sub-area from each of the axis starting at 0,0
            sub_area = array[j:j+y_edge,i:i+x_edge]
            # Store the sub-area cell numbers and shape in a dictionary
            subarea_dict[counter] = {}
            subarea_dict[counter]['sub_area'] = sub_area
            
            ## Convert the cell numbers to fingerprint positions
            subarea_dict[counter]['fp'] = cp.deepcopy(fp)
            # Iterate over x and y in the sub-area
            for a in subarea_dict[counter]['sub_area']:
                for b in a:
                    # Modify a position in an empty fingerprint from a 0 to a 1 for each cell number in sub-area
                    subarea_dict[counter]['fp'][int(b)] = 1
            counter +=1

    # Write out results for debugging
    logging.debug("original array = \n"+str(array))
    logging.debug(f'phamacophore selection size = {len(subarea_dict.keys())}')
    for i in subarea_dict.keys():
        logging.debug('subarea = \n'+str(subarea_dict[i]['sub_area']))
        logging.debug('fp = \n'+str(subarea_dict[i]['fp'])+'\n')

    return subarea_dict

def round_number(number, method = 'round'):
    if method == "ceil":
        return math.ceil(number)
    elif method == "floor":
        return math.floor(number)
    else:
        return round(number)
    
def partition_array(array, selection_size, dimension_ratio, method = 'round'):
    """
    Code to divide a given array evenly into a number of sub-areas as close to the
    selection size as possible. If the selection size cannot be fulfilled, the se-
    lection size will be decreased incrementally by one.

    Args:
        array (np.Array): numpy array that will be divided
        selection_size (int): integer determining the desired selection size
        dimension_ratio (float): the minimum ratio of short:long edge of selection area
        method(str): the method to use. from 'round', 'ceil' or 'floor'

    Returns:
        list containing numpy array objects with 1) cluster names 2) cluster numbers
    """
    
    ## Get the shape of the array
    y, x = array.shape
    
    ## Create an empty selection map
    selection_map = np.zeros((y,x),dtype='a10')
    
    ## The selection size will change if it cannot give even areas
    
    chosen_selection_size = selection_size
    
    ## This is used to track whether suitable coordinates have been chosen
    
    found_condition = False

    ## Try to find a perfect split for the data given the selection size
    ## If there is no perfect split, start reducing the selection size until splits can be found
    ## This will result in less number of selections, which will be made up with additional selections from more dense regions

    while found_condition == False:

        ## Calculate the available factors
        available_factors = find_available_dimensions(chosen_selection_size)

        ## Iterate over the factors
        for factors in available_factors:

            ## Sort the factors, largest to smalles
            factors = sorted(factors)

            if factors[0]/factors[1] > dimension_ratio:
                num_y_areas, num_x_areas = factors
                found_condition = True
                break

        ## If no suitable factors were found
        if found_condition == False:

            ## Reduce the selection size by one
            chosen_selection_size = chosen_selection_size - 1

    ## Obtain the ideal lengths of the y and x splits
    y_edge = y/num_y_areas
    x_edge = x/num_x_areas

    ## Track the coordinates which will be used
    y_split_coordinates = []
    x_split_coordinates = []

    ## Enumerate the coordinates for each split of the area
    for y_chunk in range(num_y_areas):
        y_coordinate = round_number((y_chunk+1)*y_edge, method)
        y_split_coordinates.append(y_coordinate)

    for x_chunk in range(num_x_areas):
        x_coordinate = round_number((x_chunk+1)*x_edge, method)
        x_split_coordinates.append(x_coordinate)

    ## Keep track of where the x and y splits were previously. start at zero
    curr_x = 0
    curr_y = 0

    ## Iterate over the coordinates of the splits of the arrays
    for y_ID, y_coordinate in enumerate(y_split_coordinates):
        curr_x = 0
        for x_ID, x_coordinate in enumerate(x_split_coordinates):
            
            ## Update the selection area with the cluster ID
            selection_map[curr_y:y_coordinate,curr_x:x_coordinate] = f'{x_ID,y_ID}'

            curr_x = x_coordinate

        curr_y = y_coordinate    
        
    return selection_map.astype(str), chosen_selection_size

def analyse_clusters(cluster_map, density_map):
    """
    Used to analyse clusters based on the clusters and the density.
    Creates a dictionary, keyed by cluster name which contains the densities within the cluster
    a sum of the densities and the area covered by the clusters.
    Args:
        cluster_map (np.Array): np.array object containing the clusters associated with each cell
        density_map (np.Array): np.array object containing the densities associated with each cell
    Returns:
        dict containing the densities, area and total compounds for each cluster

    """

    ## Get an array of the clusters
    clusters = np.unique(cluster_map)

    ## Start a cluster dictionary
    cluster_dict = {}

    ## Iterate over clusters
    for cluster in clusters:

        ## Ignore non-clusters
        if cluster == 0 or cluster == np.NaN or cluster == "":
            continue

        ## Save the relevant information
        cluster_dict[cluster] = {}
        cluster_dict[cluster]['densities'] = density_map[np.where(cluster_map == cluster)]
        cluster_dict[cluster]['area'] = len(cluster_dict[cluster]['densities'])
        cluster_dict[cluster]['total_density'] = cluster_dict[cluster]['densities'].sum()

    return cluster_dict

def untrim_array(trimmed_array, trimmed_number_map, untrimmed_number_map, string_type = False):
    """
    Converts the trimmed array to an untrimmed array, with the origin at true 0,0
    Args:
        trimmed_array (np.Array): a bbGAP array that has been trimmed
        trimmed_number_map (np.Array): the number mappings corresponding to the trimmed_array
        untrimmed_number_map (np.Array): the number mapping corresponding to the untrimmed array
        string_type (bool): whether the array values are strings.
    """
    if string_type:
        untrimmed_array = np.zeros(untrimmed_number_map.shape, dtype = 'a10')
    else:
        untrimmed_array = np.zeros(untrimmed_number_map.shape)

    ## Iterate over the values in the trimmed number map
    for value in np.nditer(trimmed_number_map):
        untrimmed_array[np.where(untrimmed_number_map == int(value))] = trimmed_array[np.where(trimmed_number_map == int(value))]

    return untrimmed_array

def get_fp_from_memory_mapped_db(binary_database, start_position, legnth):
    """
    Import fingerprint from a position and length in a numpy memory mapped array
    Args:
        binary_database: memory mapped numpy np.ubyte type object
        start_position (int): start position of fp in bits
        length (int): length of fingerprint in bits

    Returns:
        fingerprint in bitarray type object
    """

    # Read the array from the start position with the legnth of desired resulting array
    target_fp = binary_database[int(start_position):int(start_position+legnth)]

    # Convert to a bitarray object
    # Requires the numpy array to be converted to a list to be recognised by the bitarray module

    #target_fp = bitarray(list(target_fp))
    target_fp = bitarray(byte_to_bit(target_fp))

    return target_fp

## Extracts the reference table from the .ref file.

def create_ref_table(inref, query_fps):
    """
    Creates a pandas dataframe from the database reference file
    Also creates empty columns which will be used to record whether a compound matches any of the pharmacophore queries
    These are in the format [pharmacophore name]_[number of query fingerprint]
    Args:
        inref: input stream (opened) reference file
        query_fps (dict): query fingerprint dictionary

    Returns:
        pd.DataFrame of the reference file and query ids
    """
    ## Read in the reference file as a pandas table
    inref.seek(0)
    counter = 0
    for line in inref:
        # Count how many lines the table begins (after configuration lines)
        line = line.strip().split('\t')
        if line[0] == 'molname':
            break
        else:
            counter +=1
    logging.debug(f'skip {counter} rows')

    # Read in as csv ignoring the configuration lines
    ref_table = pd.read_csv(inref.name, sep = '\t', skiprows=counter)

    # Keep track of column names before the pharmacophore ID columns are added
    ref_table_columns = list(ref_table.columns)
    ref_table_columns.remove('molname')

    # Set the index to molname to speed up searching
    ref_table.drop_duplicates(subset='molname', keep='first', inplace=True, ignore_index=False)
    ref_table.set_index('molname',inplace = True)


    return [ref_table,ref_table_columns]



def byte_to_bit(bytes):
    """
    Convert input bytes to bits.
    Reading the file directly into the bitarray package loses a byte for some reason (bug)
    Args:
        bytes: Input as bytes
    Returns:
        bits: Input converted into bits
    """
    bits = ''.join(format(byte, '08b') for byte in bytes)
    return bits


## For bbGAP similarity

def bit_array_to_numpy_array(bitarray):
    """
    Convert a bitarray to a numpy array
    Args:
        bitarray: bitarary object
    Returns:
        np.Array: numpy array of integer 0's and 1's from bitarray
    """
    return np.array(list(bitarray.to01())).astype(np.float)

def bitarray_to_rdkit_fp(fp):
    """
    Convert a bitarray to a rdkit fingerprint
    Args:
        fp: bitarray object (fingerprint)
    Returns:
        ExplicitBitVect
    """
    return DataStructs.cDataStructs.CreateFromBitString(fp.to01())

def bleed_array(array):
    """
    Transforms the array by making each of the adjacent 8 cells to each filled cell is also filled.
    This is used to add "fuzzyness" to the similarity
    Args:
        array (np.Array): np.array containing 1s and 0s
    Returns:
        array (np.Array): where the input array has 'bled' to neighbouring cells
    """
    bleed_array = np.zeros((array.shape))
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if array[y,x] == 1:
                
                ## Check if you're at the edge of the array
                if y == array.shape[0]-1:
                    y_plus_1 = y
                else:
                    y_plus_1 = y+1
                if y == 0:
                    y_minus_1 = y
                else:
                    y_minus_1 = y-1
                if x == array.shape[1]-1:
                    x_plus_1 = x
                else:
                    x_plus_1 = x+1
                if x == 0:
                    x_minus_1 = x
                else:
                    x_minus_1 = x-1
                
                ## Update the array accordingly
                bleed_array[y,x] = 1             
                
                bleed_array[y_plus_1,x_plus_1] = 1
                bleed_array[y_minus_1,x_minus_1] = 1
                
                bleed_array[y_plus_1,x_minus_1] = 1
                bleed_array[y_minus_1,x_plus_1] = 1
                
                bleed_array[y,x_plus_1] = 1
                bleed_array[y,x_minus_1] = 1
                bleed_array[y_minus_1,x] = 1
                bleed_array[y_plus_1,x] = 1
    return bleed_array