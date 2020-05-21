import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import pandas
import traceback
from sklearn import preprocessing
    
df = pandas.read_csv('Book1.csv', header = None)
rows = len(df) * 0.7
tb_ver = rows + len(df) * .1

#df_n = df
#df = preprocessing.normalize(df)
#for col in df.columns:
   #df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

#print(rows) those for the training sets only , input and output 70% of rows
inputs = np.array(df.iloc[:int(rows),:-1].values.tolist())
outputs = df.iloc[:int(rows),-1].values.tolist()
outputs = np.array([[i] for i in outputs])

inputs = preprocessing.normalize(inputs)
outputs = preprocessing.normalize(outputs)
#inputs = preprocessing.normalize(inputs)

ver_input = np.array(df.iloc[int(rows):int(tb_ver),:-1].values.tolist())
ver_outputs = df.iloc[int(rows):int(tb_ver),-1].values.tolist()
ver_outputs = np.array([[i] for i in ver_outputs])
ver_input = preprocessing.normalize(ver_input)
ver_outputs = preprocessing.normalize(ver_outputs)

test_inputs = np.array(df.iloc[int(tb_ver):,:-1].values.tolist())
test_outputs = df.iloc[int(tb_ver):,-1].values.tolist()
test_outputs = np.array([[i] for i in test_outputs])
test_inputs = preprocessing.normalize(test_inputs)
test_outputs = preprocessing.normalize(test_outputs)

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs

        self.ver_input = ver_input
        self.ver_outputs = ver_outputs
        #initialize weights as .50 for simplicity
        self.weights = np.random.rand(19,11)*2-1
        self.weights2 = np.random.rand(11,1)*2-1

        
        self.error_history = []
        self.val_error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        try:
            if deriv == True:
                return x * (1 - x)
            return (1 / (1 + np.exp(-x)))
        except Exception:
            traceback.print_exc()

    # data will flow through the neural network.
    def feed_forward(self, val=False):
        if not val:
            self.hidden1 = self.sigmoid(np.dot(self.inputs, self.weights))
            self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights2))
            self.error  = self.outputs - self.hidden2
        else:
            self.hidden1 = self.sigmoid(np.dot(self.ver_input, self.weights))
            self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights2))
            self.error  = self.ver_outputs - self.hidden2
        
    # going backwards through the network to update weights
    def backpropagation(self):
        delta = self.error * self.sigmoid(self.hidden2, deriv=True)
        tmp_weights = np.dot(self.hidden1.T, delta)

        #self.error  = self.outputs - self.hidden2
        tmp_delta = []
        for i in delta:
            j = [elem[0] for elem in self.weights2]
            tmp_delta.append(j * i)
        delta2 = tmp_delta * self.sigmoid(self.hidden1, deriv=True)
        self.weights += 0.1*np.dot(self.inputs.T, delta2)
        self.weights2 += 0.1*tmp_weights
       

    # train the neural net for 25,000 iterations
    def train(self, epochs=0):
        #for epoch in range(epochs):
        val_error=1
        val_countToStope=0
        while(True):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            
            
            #self.epoch_list.append(epoch)
            epochs+=1

            self.epoch_list.append(epochs)
            print(F'error on {epochs} : ',np.average(np.abs(self.error)))
            if(np.average(np.abs(self.error)) < 0.3):
                break
            #if val_countToStope == 100:
            #    break
            if epochs%5 == 0:
                self.feed_forward(True)
                self.val_error_history.append(np.average(np.abs(self.error)))
                print(F'ver error on {epochs} : ',np.average(np.abs(self.error)))
                #if val_error >= np.average(np.abs(self.error)):
                #    val_error = np.average(np.abs(self.error))
                #    val_countToStope = 0
                #else:
                #    val_countToStope += 1
            else:
                # keep track of the error history over each epoch
                self.error_history.append(np.average(np.abs(self.error)))
        
            
    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        prediction2 = self.sigmoid(np.dot(prediction, self.weights2))
        return prediction2

# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

predect = []

correct = 0
total = len(test_outputs)
for i in range(len(test_inputs)):
    result = NN.predict(test_inputs[i])
    predect.append(result)
    if result >= 0.5:
        result = 1
    else:
        result = 0
    if result == test_outputs[i]:
        correct += 1
#predect_error =  test_outputs - np.array(predect)

print(f"{correct} correct predictions out of {total} that equal { round( correct/total*100, 2)}")
#print("Erorr average for predictions", np.average(np.abs(predect_error)))
#plot the error over the entire training duration
#plt.figure(figsize=(15,5))
#plt.plot(NN.epoch_list, NN.error_history)
#plt.plot(NN.epoch_list, NN.val_error_history, 'r')
#plt.xlabel('Epoch')
#plt.ylabel('Error')
#plt.show()