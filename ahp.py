from flask import Flask, request, jsonify
import numpy as np
import warnings


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
    def __init__(self, criteria=list, alternatives=list, project_name=str):
        self.project_name = project_name
        self.criteria = criteria
        self.alternatives = alternatives
        self.num_criteria = len(criteria)
        self.num_alternatives = len(alternatives)
        self.pairwise_matrix = np.ones((self.num_criteria, self.num_criteria))
        self.alternative_matrix = np.ones((self.num_criteria, self.num_alternatives, self.num_alternatives))
        self.weights = None
        self.consistency_ratio = None

    def set_pairwise_matrix(self, matrix):
        self.pairwise_matrix = np.array(matrix)

    def set_alternative_matrix(self, index, matrix):
        self.alternative_matrix[index] = np.array(matrix)

    def calculate_weights(self):
        # weights of each criteria = eigenvector of that row /sum of eigenvector elements
        eig_val, eig_vec = np.linalg.eig(self.pairwise_matrix)
        self.weights = eig_vec[:, 0] / np.sum(eig_vec[:, 0])



    def calculate_consistency_ratio(self):

        # λ_max  = weighted sum of the rows of pairwise matrix
        lambda_max = np.sum(self.weights * np.sum(self.pairwise_matrix, axis=1))

        # Consistency index = (λ_max - n)/(n-1)
        consistency_index = (lambda_max - self.num_criteria) / (self.num_criteria - 1)
        
        # This value depends on the number of criteria
        random_index = random_index_choices.get(self.num_criteria, 1.49)

        self.consistency_ratio = consistency_index / random_index



    def calculate_alternative_scores(self):
        alternative_scores = np.zeros(self.num_alternatives)
        for i in range(self.num_alternatives):
            for j in range(self.num_criteria):
                #  weights of an row (alternative) = eigen vector of the row / sum of the eigen vector elements
                eig_val, eig_vec = np.linalg.eig(self.alternative_matrix[j])
                weights = eig_vec[:, 0] / np.sum(eig_vec[:, 0])

                # Ranking Factor / Score = Inner dot product of weights of the Criteria (we got from pairwise matrix) and the weights of the Alternatives
                alternative_scores[i] += self.weights[j] * weights[i]
        return alternative_scores, eig_val, eig_vec



    def rank_alternatives(self):
        scores, eig_val, eig_vec = self.calculate_alternative_scores()
        # argsort returns the indices of the sorted form of the array (sorted array is not assigned)
        rankings = np.argsort(scores)[::-1]
        return rankings, eig_val, eig_vec


    def run(self):
        self.calculate_weights()
        self.calculate_consistency_ratio()
        rankings, eig_val, eig_vec = self.rank_alternatives()
        # Ranking the alternatives from the ranked indices
        ranked_alternatives = [self.alternatives[i] for i in rankings]
        return {
                'pairwise_matrix': self.pairwise_matrix.tolist(),
                'alternative_matrices': self.alternative_matrix.tolist(),
                'Ranking list': ranked_alternatives,
                'weights': self.weights.tolist(),
                'consistency_ratio': self.consistency_ratio,
                'eigen value':eig_val,
                'eigen vector':eig_vec
            }






# ......................... Flask API ........................

# app = Flask(__name__)

# @app.route('/ahp', methods=['POST'])
# def ahp():
#     data = request.get_json()
#     try:
#         project_name = data['name']
#         criteria = data['criteria']
#         alternatives = data['alternatives']
#         pairwise_matrix = np.array(data['pairwise_matrix'])
#         alternative_matrices = data['alternative_matrices']

#         ahp = AHP(criteria, alternatives, project_name)
#         ahp.set_criteria_pairwise_matrix(pairwise_matrix)
        
#         for criterion, matrix in alternative_matrices.items():
#             ahp.set_alternative_pairwise_matrix(criterion, np.array(matrix))
        
#         results = ahp.run()

#         sorted_rankings = sorted(results['rankings'], key=results['rankings'].get, reverse=True)

#         ranked_alternatives = {alternative: rank+1 for rank, alternative in enumerate(sorted_rankings)}

#         results['ranked_alternatives'] = ranked_alternatives

#         return jsonify(results)

#     except KeyError as e:
#         return jsonify({'error': f'Missing key in JSON data: {str(e)}'}), 400
    
#     except ValueError as e:
#         return jsonify({'error': str(e)}), 400

#     except Exception as e:
#         return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)