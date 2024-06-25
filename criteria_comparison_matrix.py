import numpy as np

def get_comparisons(no_of_criteria=int, criteria=list):
    """ 
    no_of_criteria : the number of criteria,
    criteria : list of strings of all criteria names
    """
    if no_of_criteria != len(criteria):
        raise ValueError("Number of criteria doesn't match with the number of criteria provided")
    comparisons =dict()
    for i in range(no_of_criteria):
        for j in range(i+1,no_of_criteria):
            comparisons.update({
                tuple([criteria[i], criteria[j]]) : 0 
            })
    return comparisons


def get_pairwise_comparison_matrix(comparisons=dict, no_of_criteria=int, criteria=list):
    """ 
    comparisons : (criteria,criteria) (tuple) : comparison score (float) dictionary
    no_of_criteria : the number of criteria,
    criteria : list of strings of all criteria namesa
    """
    if no_of_criteria != len(criteria):
        raise ValueError("Number of criteria doesn't match with the number of criteria provided")
    
    # pairwise comparison matrix
    pairwise_matrix = np.ones((no_of_criteria, no_of_criteria),dtype='int32').tolist()

    # indexing the criteria names
    criteria = dict(zip([x for x in range(no_of_criteria)], criteria))

    for i in range(no_of_criteria):
        for j in range(i+1,no_of_criteria):
                pairwise_matrix[i][j] = comparisons[(criteria[i],criteria[j])]
                pairwise_matrix[j][i] = 1/comparisons[(criteria[i],criteria[j])]
    
    
    return pairwise_matrix



# .............. Test ...............

# c = ['A','B','C','D','E']
# comp = get_comparisons(len(c), c)

# comp[('A', 'B')] = 8; 
# comp[('A', 'C')] = 2;
# comp[('A', 'D')] = 7;
# comp[('A', 'E')] = 5;
# comp[('B', 'C')] = 3;
# comp[('B', 'D')] = 9;
# comp[('B', 'E')] = 3;
# comp[('C', 'D')] = 7; 
# comp[('C', 'E')] = 5;
# comp[('D', 'E')] = 3;

# print(get_pairwise_comparison_matrix(comp, len(c) ,c))