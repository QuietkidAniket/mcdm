import numpy as np
#from scipy.linalg import eig
from flask import Flask, request, jsonify
from flask_cors import CORS


class AHPTOPSIS:
    def __init__(self, criteria=dict, alternatives=dict):

        
        self.criteria_names = list(criteria.keys())
        self.alternatives_names = list(alternatives.keys())

        # Define a mapping from qualitative values to numerical ones for criteria
        qualitative_to_numerical_criteria = {"bad": 0, "average": 0.5, "good": 1}

        for crit in criteria.values():
            for i in range(len(crit)):
                if type(crit[i]) == str:
                    crit[i] = qualitative_to_numerical_criteria[crit[i]] 
                elif crit[i] == None:
                    crit[i] = 0


        # Convert the qualitative criteria to numerical ones
        self.criteria = [[val for val in crit] for crit in criteria.values()]

        # Define a mapping from qualitative values to numerical ones for alternatives
        qualitative_to_numerical_alternatives = {"bad": 1, "average": 5.5, "good": 10}


        for alt in alternatives.values():
            for i in range(len(alt)):
                if type(alt[i]) == str:
                    alt[i] = qualitative_to_numerical_alternatives[alt[i]] 
                elif alt[i] == None:
                    alt[i] = 0
        # Convert the qualitative alternatives to numerical ones
        self.alternatives = [[val for val in alt] for alt in alternatives.values()]


    

    def calculate_weights(self):
        matrix = np.array(self.criteria)

        # calculating the row sum vector
        row_sum = np.sum(matrix, axis=1)

        # normalized matrix = original matrix (rows)/ row sum vector (alternatives)
        norm_matrix = matrix / row_sum[:, np.newaxis]
        
        # weights = unweighted mean of the columns/critera of normalized matrix
        weights = np.mean(norm_matrix, axis=0)

        # λ_max = weighted sum of columns / sum of columnar weights 
        lambda_max = np.sum(weights * np.sum(matrix, axis=0)) / np.sum(weights)

        # n = no. of criteria
        n = matrix.shape[0] 

        # consistency index
        ci = (lambda_max - n) / (n - 1)

        # random index
        ri = 1.98 * (n - 2) / n

        # consistency ratio
        cr = ci / ri

        err= ""

        if cr > 0.1:
            err = "Consistency ratio is greater than 0.1, please revise your criteria matrix."
            
        return matrix, norm_matrix, weights, ci, cr, err


    def calculate_rankings(self, weights):
        # Matrix of Alternative data in rows and criteria as columns
        matrix = np.array(self.alternatives)
        # Vector Normalization for each cell
        norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
        # Weighted Normalized Decision Matrix which is a result of dot product of Normalized Matrix and Criteria weight Vector 
        weighted_matrix = norm_matrix * weights

        # Ideal best and worst vectors calculated with reference to the maximum and minimum value of the criteria in the column
        ideal_best = np.max(weighted_matrix, axis=0)
        ideal_worst = np.min(weighted_matrix, axis=0)

        # Eucledian Distances from the ideal best and worst vectors to the row vectors
        s_plus = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1)); s_minus = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # ranking vector
        rankings = s_minus / (s_plus + s_minus)

        return norm_matrix, weighted_matrix, ideal_best, ideal_worst, s_plus, s_minus, rankings

    def run(self):
        matrix, norm_matrix, weights, ci, cr, err = self.calculate_weights()

        

        norm_matrix, weighted_matrix, ideal_best, ideal_worst, s_plus, s_minus, rankings  = self.calculate_rankings(weights)

        ranking_data = {name: rank for name, rank in zip(self.alternatives_names, rankings.tolist())}
        
        # # Sort the rankings dictionary by its values and get the keys
        sorted_rankings = sorted(ranking_data, key=ranking_data.get, reverse=True)

        # # Create a new rankings dictionary with ranks instead of scores
        ranked_alternatives = {alternative: rank+1 for rank, alternative in enumerate(sorted_rankings)}
        
        return {
            "Pairwise comparison matrix": matrix.tolist(),
            "Normalized pairwise comparison matrix": norm_matrix.tolist(),
            "Weights": weights.tolist(),
            "Normalized decision matrix": norm_matrix.tolist(),
            "Weighted normalized decision matrix": weighted_matrix.tolist(),
            "Ideal best": ideal_best.tolist(),
            "Ideal worst": ideal_worst.tolist(),
            "Separation from ideal best": s_plus.tolist(),
            "Separation from ideal worst": s_minus.tolist(),
            "Ranking data": ranking_data,
            "Ranking list" : ranked_alternatives,
            "Consistency Index (CI)" : ci,
            "Consistency Ration (CR)" : cr,
            "Criteria_Error" : err
        }


# ................................ Flask API ....................................

# app = Flask(__name__)
# CORS(app)

# @app.route('/ahp_topsis', methods=['POST'])
# def run_ahp_topsis():
#     data = request.get_json()
#     criteria = data.get('criteria')
#     alternatives = data.get('alternatives')

#     if not criteria or not alternatives:
#         return jsonify({'error': 'Missing criteria or alternatives'}), 400

#     ahp_topsis = AHPTOPSIS(criteria, alternatives)
#     results = ahp_topsis.run()

#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(debug=True)
    

#Example of JSON REQUEST BODY to /ahp_topsis endpoint
  #  curl -X POST -H "Content-Type: application/json" -d '{"criteria" : { "criteria1": [0.1, 0.2, 0.3],"criteria2": [0.4, 0.5, 0.6],"criteria3": [0.7, 0.8, 0.9]},"alternatives" : {"alternative1": [1, 2, 3],"alternative2": [2, 3, 4],"alternative3": [3, 4, 5]}}' http://localhost:5000/ahp_topsis



# .....................................................