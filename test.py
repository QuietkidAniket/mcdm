from ahptopisis import AHPTOPSIS
from ahp import AHP
import comparison_matrix

json_topsis  = {
    # criteria wrt criteria
    "criteria": {
        "Criterion1": ["good", "average", "bad", "good"],
        "Criterion2": ["bad", "good", "average", "bad"],
        "Criterion3": ["average", "good", "good", "average"],
        "Criterion4": ["good", "bad", "good", "bad"]
    },
    # alternatives (rows) wrt criteria (column) 
    "alternatives": {
        "Alternative1": ["good", "bad", "average", "good"],
        "Alternative2": ["bad", "bad", "good", "bad"],
        "Alternative3": ["bad", "average", "good", "average"],
        "Alternative4": ["bad", "bad", "average", "good"]
    }
}

json_ahp =  {
  "name": "Sample Model",
  "criteria": ["critA", "critB", "critC"],
  "alternatives": ["altA", "altB", "altC"],
  "pairwise_matrix": [
    [1.0, 3.0, 5.0],
    [0.333, 1.0, 2.0],
    [0.2, 0.5, 1.0]
  ],
  "alternative_matrices": {
    "critA": [
      [1.0, 0.5, 0.2],
      [2.0, 1.0, 0.333],
      [5.0, 3.0, 1.0]
    ],
    "critB": [
      [1.0, 2.0, 3.0],
      [0.5, 1.0, 2.0],
      [0.333, 0.5, 1.0]
    ],
    "critC": [
      [1.0, 0.25, 0.5],
      [4.0, 1.0, 2.0],
      [2.0, 0.5, 1.0]
    ]
  }
}

def test_topsis():

    criteria = json_topsis["criteria"]

    alternatives = json_topsis["alternatives"]

    ahp_topsis = AHPTOPSIS(criteria, alternatives)
    results = ahp_topsis.run()

    rankings = results.get('Ranking list')

    print(rankings)


def test_ahp():

    criteria = json_ahp["criteria"]

    alternatives = json_ahp["alternatives"]

    ahp = AHP(criteria, alternatives, "Supplier selection")

    ahp.set_pairwise_matrix(json_ahp["pairwise_matrix"])
    
    for i,matrix in enumerate(json_ahp["alternative_matrices"]):
        ahp.set_alternative_matrix(i,json_ahp["alternative_matrices"][matrix])


    results = ahp.run()

    rankings = results.get('Ranking list')

    print(rankings)

test_topsis()
test_ahp()

