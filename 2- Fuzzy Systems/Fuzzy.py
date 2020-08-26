import sys
import pandas
import numpy as np
from sklearn import preprocessing

df = pandas.read_csv('iris.csv', header = None)

#data normailization between 0-1
df_c = df.copy()
for col in df_c.columns:
   df_c[col] = (df_c[col]-df_c[col].min())/(df_c[col].max()-df_c[col].min())

inputs = np.array(df_c.iloc[:,:-1].values.tolist())

outputs = df.iloc[:,-1].values.tolist()
#outputs = np.array([[i] for i in outputs])

def main():

    

    #I've made a function to findd the probability each input value in the graph.
    _probably = []
    def smedium(x):
        if x >= 0 and x <= 0.6:
            return x/0.6
        if x > 0.6 and x <= 1:
            return -x/0.4 + 2.5
        else:
            return 0

    def sshort(x):
        if x >= 0 and x <= 0.6:
            return -x/0.6 + 1
        else:
            return 0

    def slong(x):
        if x > 0.6 and x <= 1:
            return x/0.4 -1.5
        else:
            return 0
            #The function tests all possibilities for each value in the array and save it in a new array.
    for i in inputs:
       for j in i:
           _probably.append([sshort(j),smedium(j),slong(j)])
            
    
    def R1(variable):
        x1,x2,x3,x4 = variable
        return(min(max(x1[0],x1[2]), max(x2[1],x2[2]), max(x3[1],x3[2]), x4[1]))

    def R2(variable):
        x1,x2,x3,x4 = variable
        return( min( max(x3[0],x3[1]), x4[0]))
    
    def R3(variable):
        x1,x2,x3,x4 = variable
        return( min( max(x2[0],x2[1]), x3[2], x4[2]))

    def R4(variable):
        x1,x2,x3,x4 = variable
        return( min( x1[1], max(x2[0],x2[1]), x3[0], x4[2]))  

    _fuzzy_output = []
    for i in range(len(_probably)):
        if i%4 ==0:
            _versicolor = max(R1(_probably[i:i+4]), R4(_probably[i:i+4]))
            _setosa = R2(_probably[i:i+4])
            _virginica = R3(_probably[i:i+4])
            _flowers = [_setosa,_versicolor,_virginica]
            _fuzzy_output.append(_flowers.index(max(_flowers))+1)
    _correct = 0
    for x in range(len(_fuzzy_output)):
        
        if _fuzzy_output[x] == outputs[x]:
            _correct +=1

    print("Correct : ",_correct/len(outputs))
    

if __name__ == '__main__':
    main()