import numpy as np
import warnings
from typing import Any
warnings.filterwarnings('ignore')


random_index_choices = {
    # number of criteria : Random Index (Saaty, 1980)
    1:0.00,
    2:0.00,
    3:0.58,
    4:0.90,
    5:1.12,
    6:1.24,
    7:1.32,
    8:1.41,
    9:1.45,
    10:1.49,
    11:1.51,
    12:1.48,
    13:1.56,
    14:1.57,
    15:1.58
}

class AHP:
    def __init__(self, criteria:list, alternatives:list, project_name:str) -> None:
        self.project_name = project_name
        self.criteria = criteria
        self.alternatives = alternatives
        self.num_criteria = len(criteria)
        self.num_alternatives = len(alternatives)
        self.pairwise_matrix = np.ones((self.num_criteria, self.num_criteria))
        self.alternative_matrices = np.ones((self.num_criteria, self.num_alternatives, self.num_alternatives))
        self.weights = None
        self.consistency_ratio = None

    def set_pairwise_matrix(self, matrix: list) -> None:
        self.pairwise_matrix = np.array(matrix)

    def set_alternative_matrix(self, index: int, matrix: list) -> None:
        self.alternative_matrices[index] = np.array(matrix)

    def calculate_weights(self) -> None:
        # weights of each criteria = eigenvector of that row /sum of eigenvector elements
        eig_val, eig_vec = np.linalg.eig(self.pairwise_matrix)
        self.weights = eig_vec[:, 0] / np.sum(eig_vec[:, 0]) 



    def calculate_consistency_ratio(self) -> None:

        # λ_max  = weighted sum of the rows of pairwise matrix
        lambda_max = np.sum(self.weights * np.sum(self.pairwise_matrix, axis=1))

        # Consistency index = (λ_max - n)/(n-1)
        consistency_index = (lambda_max - self.num_criteria) / (self.num_criteria - 1)
        
        # This value depends on the number of criteria
        random_index = random_index_choices.get(self.num_criteria, 1.49)

        self.consistency_ratio = consistency_index / random_index



    def calculate_alternative_scores(self) -> tuple[list]:
        alternative_scores = np.zeros(self.num_alternatives)
        for i in range(self.num_alternatives):
            for j in range(self.num_criteria):
                #  weights of an row (alternative) = eigen vector of the row / sum of the eigen vector elements
                eig_val, eig_vec = np.linalg.eig(self.alternative_matrices[j])
                weights = eig_vec[:, 0] / np.sum(eig_vec[:, 0])

                # Ranking Factor or Score = Inner dot product of weights of the Criteria (we got from pairwise matrix) and the weights of the Alternatives

                alternative_scores[i] += self.weights[j] * weights[i]
        return alternative_scores, eig_val, eig_vec



    def rank_alternatives(self) -> tuple[np.ndarray, float, np.ndarray]:
        scores, eig_val, eig_vec = self.calculate_alternative_scores()
        # argsort returns the indices of the sorted form of the array (sorted array is not assigned)
        rankings = np.argsort(scores)[::-1]
        return rankings, eig_val, eig_vec


    def run(self) -> dict[str, Any]:
        """ returns a json of format:
            'criteria_comparison_matrix'    : the pairwise comparison matrix of criteria
            'alternative_matrices'          : the alternative matrix
            'Ranking data'                  : the data each alternative's ranking score
            'Ranking list'                  : the ordered list of alternatives by ranks
            'weights'                       : weights of the AHP model's alternative selection process
            'consistency_ratio'             : consistency ratio
            'eigen value'                   : eigen value of the associated criteria matrix
            'eigen vector'                  : eigen vector of the associated criteria matrix's rows
        """
        self.calculate_weights()
        self.calculate_consistency_ratio()
        rankings, eig_val, eig_vec = self.rank_alternatives()
        # Ranking the alternatives from the ranked indices
        ranked_alternatives = [self.alternatives[i] for i in rankings]
        return {
                'criteria_comparison_matrix': self.pairwise_matrix.tolist(),
                'alternative_matrices': self.alternative_matrices.tolist(),
                'Ranking data': rankings.tolist(),
                'Ranking list': ranked_alternatives,
                'weights': self.weights.tolist(),
                'consistency_ratio': self.consistency_ratio,
                'eigen value':eig_val,
                'eigen vector':eig_vec.tolist()
            }

