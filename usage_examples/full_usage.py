import sys
import os
sys.path.append('../')
from bbSelectBuild import bbSelectBuild
from bbSelect import Picker
import logging

logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.INFO)


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15,10]

def main():
    smiles_file = './data/enamine_acids/enamine_acids_filtered.csv'
    output_root = './data/enamine_acids/enamine_acids_full_usage_example_py'
    cell_size = 1
    num_cells = 20
    ncpu = len(os.sched_getaffinity(0)) -1

    # Set the SMARTS that will be used to clip the "core scaffold" to [15CH3]. 
    # If using the results from an R-group decomposition, you need to generate the clipped smiles file yourself replacing * with [15CH3]

    rxn_smarts = '[C:1](=[O:2])[OH]>>[C:1](=[O:2])[15CH3]'

    # Below are the default settings for bbSelectBuild and do not need to be set if using default. 
    # However, if changing anything, these will need to be set

    conformer_generation_program = 'omega' # Can be set to 'rdkit' if rdkit is to be used. Warning: this takes MUCH longer

    # These are required to run omega from python
    omega_load_command = 'module load openeye'
    omega_run_command = 'omega2'

    # If the clipped smiles file has already been generated you can load that in and define the separator in the file
    # If you are using the results of an R-group decomposition, the clipped file will need to be loaded in where * has been replaced by [15CH3]
    clipped_smiles_file  = None
    clipped_smiles_file_sep = None

    # If the conformers, or the aligned conformers have already been generated, set the paths here
    conformer_sdf = None
    aligned_mols_sdf = None

    # If you would like to save the aligned conformers, you can set the path here
    save_aligned_mols_sdf = None
    
    bbSelectBuilder = bbSelectBuild(
                                smiles_file = smiles_file, 
                                output_root = output_root, 
                                cell_size = cell_size, 
                                num_cells = num_cells, 
                                ncpu = ncpu, 
                                rxn_smarts = rxn_smarts
                                )

    # Set options in bbSelectBuild
    bbSelectBuilder.SetOption('conformer_generation_program', conformer_generation_program)
    bbSelectBuilder.SetOption('omega_load_command', omega_load_command)
    bbSelectBuilder.SetOption('omega_run_command', omega_run_command)
    bbSelectBuilder.SetOption('clipped_smiles_file', clipped_smiles_file)
    bbSelectBuilder.SetOption('clipped_smiles_file_sep', clipped_smiles_file_sep)
    bbSelectBuilder.SetOption('conformer_sdf', conformer_sdf)
    bbSelectBuilder.SetOption('aligned_mols_sdf', aligned_mols_sdf)
    bbSelectBuilder.SetOption('save_aligned_mols_sdf', save_aligned_mols_sdf)
    
    bbSelect = bbSelectBuilder.Run()
    bin_file = bbSelect.GetBinLocation()
    ref_file = bbSelect.GetRefLocation()
    print(bin_file, ref_file)
    
    n_select = 48
    ncpu = len(os.sched_getaffinity(0)) -1
    sort = 'MPO'
    tanimoto = 0.9
    pharmacophores = '*'
    method = 'classic'
    use_coverage = False
    
    bbSelection = Picker(ref_file = ref_file, 
                            bin_file = bin_file, 
                                   n = n_select,  # How many compounds to select
                              method = 'som', # Which clustering method to use
                      pharmacophores = pharmacophores, # Which pharmacophore to select from,
                                ncpu = ncpu,  # Number of cpus to use,
                                sort = sort,
                                tanimoto = tanimoto,
                                use_coverage = use_coverage)

    img = bbSelection.DrawSelectedMols(align_smiles = 'CC(=O)O')
    with open('output.svg', 'w') as f_handle:
            f_handle.write(img.data)
            
if __name__ == '__main__':
    main()
