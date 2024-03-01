# bbSelect
**A package for diverse R-group selection for medicinal chemistry**

**bbSelect** is a tool for selecting a diverse set of R-groups by using information about the placement of pharmacophore features in 3D. It looks to map the available pharmacophore feature placement space in 3D, projecting this to a 2D representation which is then used to drive a partitioning and selection.

## Installation instructions
Mamba can be used to install the enviroment needed to run **bbSelect** using the included environment.yml file.

The following commands can be used to install from fresh from linux, setting up a fresh miniforge (https://github.com/conda-forge/miniforge)
```
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3.sh -b -p "${HOME}/conda"

source "${HOME}/conda/etc/profile.d/conda.sh"

source "${HOME}/conda/etc/profile.d/mamba.sh"

mamba env create -f environment.yml
```

## Usage

>[!TIP]
>Filtering **must** reduce the input set to molecules that would be permissable to select.

The Usage for **bbSelect** is exemplified in the usage_examples directory.\
Its use is demonstrated within jupyter notebooks but can be wrapped up into a python script to run in a more automated fashion.\
The general steps for the use of bbSelect are:

<p align="center"><b>
<br>Get library of molecules<br>
🠋<br>
Filter molecules and associate with properties for prioritisation</b><i><br> (See usage_examples/bb_preparation.ipynb)<br><b></i>
🠋<br>
Clip molecules or perform R-group decomposition, placing 15CH3 at the attachment point </b><i><br>(Supported in bbSelectBuild.py)<br><b></i>
🠋<br>
Generate conformer ensembles </b><i><br>(Supported in bbSelectBuild.py)<br><b></i>
🠋<br>
Build bbSelect DB </b><i><br>(bbSelectBuild.py)<br><b></i>
🠋<br>
Partition and select R-groups </b><i><br>(bbSelect.py)<br><br><b></i>
</b></p>

>[!WARNING]
>Prioritisation **must** be performed elsewise there is no method for **bbSelect** to determine a selection from a partition.\
>It is recommended to use composite scores (such as MPOs, QED).\
>**bbSelect** selects from the prioritisation values in ascending order, where smallest values are chosen first.

## Settings
Described herein are the main settings for bbSelectBuild and bbSelect\

### bbSelectBuild 
The functions of bbSelect build are wrapped into the self-titled bbSelectBuild class.\
**Main parameters for bbSelectBuild.bbSelectBuild()**\

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| smiles_file | str | - | Path to the smiles file of unclipped molecules, containing the headers `smiles` and `ID` |
| output_root | str | - | Path to desired output file. The bbSelect db will append `.bin` and `.ref` to this root to store the db files |
| cell_size | float | - | The size of the cells in the grid, in Angstroms |
| num_cells | int | - | The number of cells, in the positive x direction, to use in the grid |
| ncpu | int | `0` | Number of CPUs to use in the calculations. Setting to `0` will use all available CPUs |
| rxn_SMARTS | str | - | SMARTS pattern that will transform the full molecule to its clipped representation with [15CH3] at the attachment point|

**Options for bbSelectBuild class using bbSelectBuild.SetOption(option, value)**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| conformer_generation_program | str | `'omega'` | `'rdkit'` or `'omega'` |
| omega_load_command | str | `module load openeye` | bash command used to load the omega program |
| omega_run_command | str | `omega2` | bash command to run the omega program |
| clipped_smiles_file | str | - | Path to clipped smiles file if it has been generated |
| clipped_smiles_file_sep | str | `\t` | Delimiter used in clipped smiles file |
| conformer_sdf | str | - | Path to conformer SDF if already generated|
| aligned_mols_sdf | str | - | Path to aligned conformer SDF if already generated|
| save_aligned_mols_sdf | str | - | Path to store aligned conformer SDF if desired|

>[!WARNING]
>Conformers must be generated on the clipped R-groups, with [15CH3] at the attachment point

### bbSelect
The functions of bbSelect are wrapped into the Picker() class.\
**Main parameters for bbSelect.Picker()**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| ref_file | str | -| Path to the .ref bbSelect db file |
| bin_file | str | -| Path to the .bin bbSelect db file |
| n | int | - | Number of compounds to select |
| method | str | `classic` | `som`, `classic`, or `full_coverage` |
| pharmacophores | str | `*` | Comma separated list of pharmacophore to select from. `*` selects from all available |
| ncpu | str | - | Nuumber of CPUs to utilise|
| sort | str | - | Which value from the input smiles to use for prioritisation. Multiple sort orders can be configured with a comma-separated list|
| tanimoto | float | `0.9` | Tanimoto threshold for selections. Set to `0` to skip this|
| use_coverage | bool | `False` | Whether to force that selections increase coverage|
| flat_som | bool | `False` | Whether to transform all values to the SOM as 1. When False, values will be transformed to log base2|
| multiplier | float | `1` | Multiplier to transform values to the SOM|


## Acknowledgements
Special thank you to Stephen Pickett, Peter Pogany, David Palmer, Nick Tomkinson, and Darren Green for their support during the development of this code\
Marcus Farmer for the code used to create the MPOs\
[JustGlowing minisom](https://github.com/JustGlowing/minisom/) open-source SOM package used and included in this code


## Zipped files
In order to run the code as given in the usage examples, the following files need to be unzipped

`./data/enamine_acids/rdkit_conformers/enamine_acids_filtered_aligned_rdkit.zip`\
`./data/enamine_acids/enamine_acids_filtered_clipped_omega.zip`\
`./data/enamine_acids/enamine_acids_filtered_aligned.zip`\
`./data/enamine_acids/enamine_acids_filtered_clipped_conformers.zip`\
`./data/enamine_acids/rdkit_conformers/enamine_acids_filtered_clipped_rdkit.zip`

