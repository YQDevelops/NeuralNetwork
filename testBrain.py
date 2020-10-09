#############################################################################################################################################################################################################################################################################################################################################################################################################################################################################
#
#       By:Low Yew Qing
# Objective:To Test the Neural Network That Was Created in neuralnetwork.py and Implementing The User Interface
#    Edits:29/03/2020--Added a way to see accuracy of the network. There's a bug because the prediction is always the same.
#          01/04/2020--Moved the accuracy tester to neuralnetwork.py based on Sebastian Lague's tutorial.
#          03/04/2020--Tried testing the cost function. Took to long to run over the training images.
#          10/09/2020--Made changes in accordance to PEP8
##############################################################################################################################################################################################################################################################################################################################################################################################################################################################################
import neuralnetwork as nn
import numpy as np

with np.load("mnist.npz") as data:
    training_images = data["training_images"]  # 50000 images
    training_labels = data["training_labels"]  # 50000 labels.

layer_sizes = (784, 5, 10)

brain = nn.ArtificialBrain(layer_sizes)
print(brain.weights)
prediction = brain.predict(training_images)
print(prediction)
# print(brain.getCost(training_images,training_labels))
brain.print_accuracy(training_images, training_labels)
