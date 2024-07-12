import torch
from typing import Any


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

class AHPTOPSIS:
    def __init__(self, criteria:dict, alternatives:dict):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")    
        
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


    

    def calculate_weights(self)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, str]:
        matrix = torch.Tensor(self.criteria, device = self.device)


        # calculating the row sum vector
        row_sum = torch.sum(matrix, axis=1)
        row_sum.unsqueeze_(1)

        # normalized matrix = original matrix (rows)/ row sum vector (alternatives)
        norm_matrix = matrix / row_sum
        
        # weights = unweighted mean of the columns/critera of normalized matrix
        weights = torch.mean(norm_matrix, axis=0)

        # λ_max = weighted sum of columns / sum of columnar weights 
        lambda_max = torch.dot(weights , torch.sum(matrix, axis=0)) / torch.sum(weights)

        # n = no. of criteria
        n = matrix.shape[0] 

        # consistency index
        ci = (lambda_max - n) / (n - 1)

        # random index
        ri  = random_index_choices.get( len(self.criteria_names), 1.49)

        # consistency ratio
        cr = ci / ri

        err= ""

        if cr > 0.1:
            err = "Consistency ratio is greater than 0.1, please revise your criteria matrix."
            
        return matrix, norm_matrix, weights, float(ci), float(cr), err


    def calculate_rankings(self, weights: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Matrix of Alternative data in rows and criteria as columns
        matrix = torch.Tensor(self.alternatives, device = self.device)
        # Vector Normalization for each cell
        norm_matrix = matrix / torch.sqrt(torch.sum(matrix**2, axis=0))

        # Weighted Normalized Decision Matrix which is a result of dot product of Normalized Matrix and Criteria weight Vector 
        weighted_matrix = torch.mul(weights, norm_matrix)

        # Ideal best and worst vectors calculated with reference to the maximum and minimum value of the criteria in the column
        ideal_best = torch.max(weighted_matrix, axis=0).values
        ideal_worst = torch.min(weighted_matrix, axis=0).values

        # Eucledian Distances from the ideal best and worst vectors to the row vectors
        s_plus = torch.sqrt(torch.sum((weighted_matrix - ideal_best)**2, axis=1)); s_minus = torch.sqrt(torch.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # ranking vector
        rankings = s_minus / (s_plus + s_minus)

        return norm_matrix, weighted_matrix, ideal_best, ideal_worst, s_plus, s_minus, rankings

    
    
    def run(self) -> dict[str, Any]:
        matrix, norm_matrix, weights, ci, cr, err = self.calculate_weights()

        norm_matrix, weighted_matrix, ideal_best, ideal_worst, s_plus, s_minus, rankings  = self.calculate_rankings(weights)

        ranking_data = {name: rank for name, rank in zip(self.alternatives_names, rankings.tolist())}
        
        # Sort the rankings dictionary by its values and get the keys
        sorted_rankings = sorted(ranking_data, key=ranking_data.get, reverse=True)

        # Create a new rankings dictionary with ranks instead of scores
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
