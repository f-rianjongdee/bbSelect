import sys
import logging
import matplotlib.pyplot as plt
import random
import numpy as np
from rdkit import Chem, SimDivFilters, DataStructs
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
sys.path.append('../')
#from bbSelectBuild import bbSelectBuild
from bbSelect import Picker
import logging
logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.ERROR)



import warnings
warnings.filterwarnings("ignore")

def main():
    

    ncpu = 4


    bin_file = '../data/enamine_acids/enamine_acids/enamine_acids_clogp.bin'
    ref_file = '../data/enamine_acids/enamine_acids/enamine_acids_clogp.ref'


    start_selection = Picker(ref_file = ref_file, 
                        bin_file = bin_file, 
                               n = ncpu,  # How many compounds to select
                          method = 'som', # Which clustering method to use
                  pharmacophores = '*', # Which pharmacophore to select from,
                            ncpu = ncpu,  # Number of cpus to use,
                            sort = 'MPO',
                            tanimoto = 0.9)
    
    smiles = start_selection.GetAllSmiles()
    molnames = start_selection.GetAllIds()
    data_table_MPO = start_selection.GetDataTable()['MPO']
    
    mols = []
    
    for sm, ID in zip(smiles, molnames):
        #try:
        mol = Chem.MolFromSmiles(sm)
        #except:
            #print(f'could not convert {sm};{ID}')

            #continue
        mols.append(mol)
        
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m,2,2048) for m in mols]
    
    
    ## Default parameters
    sort = 'MPO'
    tanimoto = 0.9
    pharmacophores = '*'

    ## Parameters to search

    n_selects = [12,24,36,48,60,72,96,128,160,200,250]
    methods = ['som','classic','random','maxMin','sphex','kMeans']
    use_coverages = [True, False]

    ## Save file
    with open('coverage_testing_all_methods1.csv', 'w') as out_file:

        first_line = True

        for n_select in n_selects:

            for method in methods:
                
                for use_coverage in use_coverages:

                    if method not in ['som','classic'] and use_coverages == False:
                        
                        continue
                    
                    elif method in ['som','classic']:
                        
                        bbSelection = Picker(

                            ref_file = ref_file, 
                            bin_file = bin_file, 
                                   n = n_select,  # How many compounds to select
                              method = method, # Which clustering method to use
                      pharmacophores = pharmacophores, # Which pharmacophore to select from,
                                ncpu = ncpu,  # Number of cpus to use,
                                sort = sort,
                                tanimoto = tanimoto,
                                use_coverage = use_coverage)

                        selected_molnames = bbSelection.GetSelectionIDs()
                        coverage = bbSelection._coverage
                        MPO_values = data_table_MPO.loc[selected_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        n_selected = n_select


                    # RANDOM ###############################################
                    elif method == 'random':

                        ## Get random selection
                        random.seed(3)
                        random_picks = random.sample([i for i,j in enumerate(mols)], n_select)
                        random_molnames = [molnames[x] for x in random_picks]
                        coverage = start_selection.SimulateCoverage(molnames = random_molnames)
                        MPO_values = data_table_MPO.loc[random_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        n_selected = n_select


                    # MaxMin ###############################################  
                    elif method == 'maxMin':
                        ## Start MaxMinPicker class
                        mmp =SimDivFilters.MaxMinPicker()

                        def fn(i,j,fps = fps):
                            return 1.-DataStructs.TanimotoSimilarity(fps[i],fps[j])

                        MaxMin_picks = mmp.LazyPick(fn,len(fps), n_select, seed=3)
                        MaxMin_molnames = [molnames[x] for x in MaxMin_picks]
                        coverage = start_selection.SimulateCoverage(molnames = MaxMin_molnames)
                        MPO_values = data_table_MPO.loc[MaxMin_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        n_selected = n_select

                    # Sphere Exclusion ###############################################
                    elif method == 'sphex':
                        ## Calculate fingerprints
                        ## Equivalent to ECFP4 (diameter used in ECFP, radius used in Morgan)

                        # Because sphex uses tanimoto, using values between 0.85 and 0.72 which give similar range
                        sphex_dict = {12:0.85, 24:0.82, 36:0.8, 48:0.79, 60:0.78, 72:0.77, 96:0.76, 128:0.75, 160:0.74, 200:0.73, 250:0.72}
                        thresh = sphex_dict[n_select]

                        ## Start LeaderPicker object. This performs a fast sphere exclusion clustering without calculating similarity matrix
                        lp = rdSimDivPickers.LeaderPicker()

                        ## Get 'picks' which are essentially the cluster centroids, giving the indices in the list of molecules of the selected compounds
                        sphex_picks = lp.LazyBitVectorPick(fps,len(fps),thresh)
                        sphex_molnames = [molnames[x] for x in sphex_picks]
                        coverage = start_selection.SimulateCoverage(molnames = sphex_molnames)
                        MPO_values = data_table_MPO.loc[sphex_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        n_selected = len(sphex_picks)

                    # k Means ###############################################
                    elif method == 'kMeans':

                        ## Convert fingerprints into a list
                        fps_list = [list(fp) for fp in fps]

                        ## Set the selection size e.g. number of k-means clusters
                        num_clusters = n_select

                        ## Initialise the k-means clustering object
                        km = KMeans(
                            n_clusters=num_clusters, init='k-means++',
                            n_init=10, max_iter=300, 
                            tol=1e-04, random_state=3)

                        ## Perform k-means clustering
                        y_km = km.fit(fps_list)

                        ## Number of iterations. If less that max_iter, convergence has been met
                        # y_km.n_iter_

                        ## Get the clusters and the values of the centroids
                        m_clusters = y_km.labels_.tolist()
                        centers = np.array(y_km.cluster_centers_)

                        ## Find the data point closest to the cluster centroid
                        k_means_molnames = []
                        k_means_picks = [] ## Indices of selection

                        ## Iterate over the clusters

                        for i in range(num_clusters):
                            ## Get the centroid vector
                            center_vec = centers[i]

                            ## Get the indices of the data points that are within each cluster
                            data_idx_within_i_cluster = [ idx for idx, cluster_num in enumerate(m_clusters) if cluster_num == i ]

                            ## Create a matrix where the number of rows = number of points in cluster and is the legnth of the data vector (length of fingerprint)
                            ## This is a 'slice' of the data within the original data set which come from the cluster
                            one_cluster_tf_matrix = np.zeros( (  len(data_idx_within_i_cluster) , centers.shape[1] ) )

                            ## Iterate over the compounds within the cluster
                            for row_num, data_idx in enumerate(data_idx_within_i_cluster):

                                ## Get the fingerprint for the compound
                                one_row = fps_list[data_idx]

                                ## Set the fingerprint within the matrix
                                one_cluster_tf_matrix[row_num] = one_row

                            ## Calculate the closest compound to the centroid vector
                            closest, distance = pairwise_distances_argmin_min(center_vec.reshape(1, -1), one_cluster_tf_matrix)

                            ## Get the index of the closest compound to the centroid vector in the slice cluster list
                            closest_idx_in_one_cluster_tf_matrix = closest[0]

                            ## Get the index of the closest compound in the larger data set
                            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]

                            ## Get the molname
                            data_id = molnames[closest_data_row_num]

                            k_means_picks.append(closest_data_row_num)
                            k_means_molnames.append(data_id)

                        k_means_molnames = list(set(k_means_molnames))

                        coverage = start_selection.SimulateCoverage(molnames = k_means_molnames)
                        MPO_values = data_table_MPO.loc[k_means_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        n_selected = n_select

                    ## Write output!
                    if first_line == True:
                        header_line = f'n_select,method,use_coverage,{",".join(list(coverage.keys()))},MPO_mean,MPO_std\n'
                        out_file.write(header_line)
                        print(header_line)
                        first_line = False

                    output_line = f'{n_selected},{method},{use_coverage},{",".join([str(coverage[x]) for x in coverage.keys()])},{MPO_mean},{MPO_std}\n'
                    out_file.write(output_line)  
                    print(output_line)

if __name__ == '__main__':
    main()