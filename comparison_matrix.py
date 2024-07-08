import numpy as np

def get_comparisons(n:int, items:list) -> dict[tuple[str, str], int]:
    """ 
    n : the number of items,
    criteria : list of strings of all item names
    """
    if n != len(items):
        raise ValueError("Number of items doesn't match with the number of criteria provided")
    comparisons =dict()
    for i in range(n):
        for j in range(i+1,n):
            comparisons.update({
                tuple([items[i], items[j]]) : 0 
            })
    return comparisons


def get_pairwise_comparison_matrix(comparisons:dict[tuple[str, str], int], n:int, items:list):
    """ `
    comparisons : (item,item) (tuple) : comparison score (float) dictionary
    n : the number of items,
    criteria : list of strings of all item namesa
    """
    if n != len(items):
        raise ValueError("Number of items doesn't match with the number of criteria provided")
    
    # pairwise comparison matrix
    pairwise_matrix = np.ones((n, n),dtype='int32').tolist()

    # indexing the item names
    items = dict(zip([x for x in range(n)], items))

    for i in range(n):
        for j in range(i+1,n):
                pairwise_matrix[i][j] = comparisons[(items[i],items[j])]
                pairwise_matrix[j][i] = 1/comparisons[(items[i],items[j])]
    
    
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