import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def compute_composite(probability_list, weight_list):
    for i in range(len(probability_list)):
        prob = probability_list[i]
        if prob[0] == False:
            weight_list[i] = -1
    print(weight_list)

    reweight_factor = 0
    for i in range(len(weight_list)):
        weight = weight_list[i]
        if weight != -1:
            reweight_factor += weight

    weight_list = [weight/reweight_factor for weight in weight_list]
    print(weight_list)

    composite_score = 0
    for i in range(len(probability_list)):
        prob = probability_list[i]
        if prob[0] == True:
            composite_score += weight_list[i] * prob[1]
    print(composite_score)

    return sigmoid(composite_score)

probability_list = [(True, 0.9), (True, 0.2), (False, -1)]
weight_list = [0.4, 0.3, 0.3]
composite_score = compute_composite(probability_list, weight_list)

print(composite_score)