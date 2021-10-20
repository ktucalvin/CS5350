from id3 import ID3

attributes = {
  "color": ["white", "black"],
  "shape": ["triangle", "circle", "square"]
}
labels = ["+", "-"]
S = []
with open("./datasets/weighted_id3/train.csv") as file:
  for line in file:
    S.append(tuple(line.strip().split(',')))

print(S)

unweighted_model = ID3()
unweighted_model.train(S, attributes, labels, max_depth=1)
print(unweighted_model.tree)

# first 6 examples should select color
w1 = 6*[1e12] + 6*[1e-12]

# last 6 examples should select shape
w2 = 6*[1e-12] + 6*[1e12]

color_model = ID3()
color_model.train(S, attributes, labels, weights=w1, max_depth=1)
print(color_model.tree)

shape_model = ID3()
shape_model.train(S, attributes, labels, weights=w2, max_depth=1)
print(shape_model.tree)
