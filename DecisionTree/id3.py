# p_i is probability of i-th label
# p_i = count(i-th label) / len(labels)
# ex: binary, p_+ and p_- are probability of positive and negative labels

class ID3:
    def entropy(S):
        pass
        # - sum_i^K p_i log_2(p_i)

    def majority_error(S):
        pass
        # determine majority label
        # return count(minority labels) / count(labels)

    def gini_index(S):
        pass
        # see top-level note for what p_k is
        # return 1 - sum_1^K p_k^2

    def gain(S, A, measure):
        pass
        # return measure(S) - sum_{v in values(A)} (len(S_v) / len(S) measure(S_v))

    # :NOTE: "label is the target attribute (prediction)"
    def train(S, attributes, label, measure=entropy, max_depth=float("inf"), depth = 0):
        pass
        if depth > max_depth: return
        # if all examples have same label
        #   return a leaf node with that label
        # if attributes is empty
        #   return a leaf node with the most common label

        # create root node for tree
        # A = attribute in attributes that best splits S
        # A = max([gain(S, attr) for attr in attributes])
        # for each value v that A can take:
        #   add a new tree branch corresponding to A=v
        #   let S_v be the subset of examples in S with A = v
        #   if S_v is empty:
        #       add leaf node with the most common label in S
        #   else:
        #       below this branch add the subtree ID3(S_v, attributes - {A}, label)
        # return root node

    def predict(tree, input):
        pass


if __name__ == "__main__":
    """ When run from CLI, just dump data necessary to do hw1 """
    # data-desc is not standardized data has to be pre-processed

    # Car example
    car_models = [ID3() for _ in range(6)]
    car_data = []
    car_attributes = {
        "buying":   ["vhigh", "high", "med", "low"],
        "maint":    ["vhigh", "high", "med", "low"],
        "doors":    ["2", "3", "4", "5more"],
        "persons":  ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety":   ["low", "med", "high"]
    }
    labels = ["unacc", "acc", "good", "vgood"]
    with open("./datasets/car/train.csv") as file:
        for line in file:
            [buying,maint,doors,persons,lug_boot,safety,label] = line.strip().split(',')
            car_data.append((buying,maint,doors,persons,lug_boot,safety,label))

    car_val = []
    car_predictions = []
    with open("./datasets/car/test.csv") as file:
        for line in file:
            [buying,maint,doors,persons,lug_boot,safety,label] = line.strip().split(',')
            car_val.append((buying,maint,doors,persons,lug_boot,safety,label))
    
    for depth, model in enumerate(car_models, start=1):
        model.train(car_data, car_attributes, measure=ID3.entropy, max_depth=depth)
    
    for i in range(6):
        car_predictions.append([model.predict(input) for input in car_val])
    
    print(car_predictions)
