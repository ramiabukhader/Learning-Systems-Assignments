import math
import numpy as np
#I OVA MI TREBA KLASA 

class City:
    def __init__(self, name, x,y):
        self.name=name
        self.x=x
        self.y=y

class Cities: 
    def __init__(self,cities):
        self.distance_table=np.zeros([len(cities),len(cities)]) #matrica da ne racunam distance uvek        
        self.cities=cities
        

    def Fill_table(self):
        for city1 in self.cities:
            for city2 in self.cities:
                if(city1.name==city2.name):
                    break
                self.distance_table[city1.name-1][city2.name-1]=Distance_two_city(city1,city2)
                self.distance_table[city2.name-1][city1.name-1]=Distance_two_city(city1,city2)


 


def Distance_two_city(p1, p2):    
    dist=math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)
    return dist        
    
def ReadFile():
    elements=[]
    f=open('input.tsp','r')

    f.readline() #name
    f.readline() #typeFile
    f.readline() #commentar
    f.readline() #dimension
    f.readline()  #edgeWeightType
    f.readline()
        
    line=f.readline()
        
    while('EOF' not in line):
        words=line.replace('\t',' ').split(' ')
            # delete \n in words
        words[2]=words[2][0:len(words[2])-1]
        el=City(int(words[0]),float(words[1]),float(words[2]))
        elements.append(el)
        line=f.readline()


    f.close()
            

    return elements

        
def CreateCities(): #0vu fju pozovi
    cities=Cities(ReadFile())
    cities.Fill_table()

    return cities



    

