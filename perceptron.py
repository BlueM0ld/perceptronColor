import webcolors
import math

#intialize weights
weight0 = 0.0
weight1 = 0.0
weight2 = 0.0
biasWeight = 0.0


epoch = 1000

#training input
traininginputs = []
traininginputs.append([144,238,144]) #lightgreen # black
traininginputs.append([255,255,224]) #lightyellow # black
traininginputs.append([0,100,0]) #darkgreen # white
traininginputs.append([85,107,47]) #darkolivegreen # white
traininginputs.append([0,0,0]) #darkolivegreen # white
traininginputs.append([0,0,255]) #blue # white
traininginputs.append([216,191,216])#thistle # black
traininginputs.append([88,143,143])#rosybrown # black

#expected output of the training inputs
tiTrueOutputs = [1,1,0,0,0,0,1,1]

#activation function --- function
def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))


def train():
    learningRate = 0.1
    global weight0 
    global weight1
    global weight2
    global biasWeight
    
    for _ in range(epoch):
        for inputs, tiTrueOutput in zip(traininginputs, tiTrueOutputs):
             prediction = threshold_frequency_calc(inputs)
             weight0 += learningRate * (tiTrueOutput - prediction) * inputs[0]
             weight1 += learningRate * (tiTrueOutput - prediction) * inputs[1]
             weight2 += learningRate * (tiTrueOutput - prediction) * inputs[2]
             biasWeight += learningRate * (tiTrueOutput - prediction)


def threshold_frequency_calc(inputs):

    
    inputWeightR = inputs[0] * weight0
    inputWeightG = inputs[1] * weight1
    inputWeightB = inputs[2] * weight2

    # combinedInputWeights / the aggregated value
    x = inputWeightR + inputWeightB + inputWeightG + biasWeight

    activationFunction = sigmoid(x)
    if activationFunction >0.5:
        activationFunction = 1 #black
    else:
        activationFunction = 0 #white
    
    return activationFunction




train()
val = input("Enter colour input: ")
RGB = webcolors.name_to_rgb(val)

print(RGB)

textColor = threshold_frequency_calc(RGB)

if textColor ==1:
      print('black')
else:
      print('white')

