import sys
sys.path.append('../')
#from bbSelectBuild import bbSelectBuild
from bbSelect import Picker
import logging
logging.basicConfig(format = '%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level = logging.ERROR)

def all_as_one(value):
    return 1

import warnings
warnings.filterwarnings("ignore")

def main():
    
    bin_file = '../data/enamine_acids/enamine_acids/enamine_acids_clogp.bin'
    ref_file = '../data/enamine_acids/enamine_acids/enamine_acids_clogp.ref'


    ## Default parameters
    ncpu = 4
    sort = 'MPO'
    tanimoto = 0.9
    pharmacophores = '*'

    ## Parameters to search

    n_selects = [12,24,36,48,60,72,96,128,200,300]
    methods = ['som','classic']
    flat_soms = [True, False]
    use_coverages = [True, False]

    ## Save file
    with open('coverage_testing_bbSelect_parameters1.csv', 'w') as out_file:

        first_line = True

        for n_select in n_selects:

            for method in methods:

                for use_coverage in use_coverages:

                    for flat_som in flat_soms:

                        if flat_som == True and method == 'classic':

                            continue

                        else:
                            func = None

                        bbSelection = Picker(

                            ref_file = ref_file, 
                            bin_file = bin_file, 
                                   n = n_select,  # How many compounds to select
                              method = method, # Which clustering method to use
                      pharmacophores = pharmacophores, # Which pharmacophore to select from,
                                ncpu = ncpu,  # Number of cpus to use,
                                sort = sort,
                                tanimoto = tanimoto,
                                use_coverage = use_coverage,
                        flat_som = flat_som)

                        coverage = bbSelection._coverage
                        selected_molnames = bbSelection.GetSelectionIDs()
                        
                        data_table_MPO = bbSelection.GetDataTable()['MPO']
                        MPO_values = data_table_MPO.loc[selected_molnames]
                        MPO_mean = MPO_values.mean()
                        MPO_std = MPO_values.std()
                        
                        
                        
                        if first_line == True:
                            header_line = f'n_select,method,use_coverage,flat_som,{",".join(list(coverage.keys()))},MPO_mean,MPO_std\n'
                            out_file.write(header_line)
                            print(header_line)
                            first_line = False

                        output_line = f'{n_select},{method},{use_coverage},{flat_som},{",".join([str(coverage[x]) for x in coverage.keys()])},{MPO_mean},{MPO_std}\n'
                        out_file.write(output_line)  
                        print(output_line)

    print(f'finished')
    
            
if __name__ == '__main__':
    main()