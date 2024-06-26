# Supplier Selection

NOTE : THIS PORTION IS JUST A QUICK CHEATSHEET.    

The parameters and return type of all the functions have not been discussed here. You can look those up via hovering over the functions for seeing their declarations. 

## Analytic Hierarchical Process (AHP)

ahp.py contains class ```AHP``` for Analytic Hierarchical Process

use the following steps to run AHP:

*   >from ahp import AHP
    >import criteria_comparison_matrix
    

* create an AHP object using ```AHP(criteria, alternative, project_name)```

* call the ```criteria_comparison_matrix.get_comparisons(no_of_criteria, criteria_list)``` to get a dictionary of criteria comparisons which shall serve as a template for the pairwise comparison matrix.

* call the ```criteria_comparison_matrix.get_pairwise_comparison_matrix(comparisons, no_of_criteria, criteria_list)``` to get the pairwise comparison matrix in form of a list.

* call the ```AHP.set_pairwise_matrix(matrix)``` to set the pairwise comparison matrix 

* call the ```AHP.set_alternative_matrix(matrix)``` to set the alternative comparison matrix

* call the ```AHP.run()``` to run the AHP algorithm and get the required results in json format.


## Topsis - Technique 

*   > from ahptopsis import AHPTOPSIS

* create a topsis object using ```AHPTOPSIS(criteria(dict), alternative(dict))```

* call the AHPTOPSIS.run() function to run the TOPSIS algorithm and get the required results in json format.
