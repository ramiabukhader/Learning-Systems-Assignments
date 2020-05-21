#GeneticAlgorithm versija 2
from ausxiliaryAlgorithms import Distance
#from random import randrange,choice
import random
from cities import Cities,CreateCities
import math
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

SAVE_BEST_PERCENT=10 #elitism , crossover rate=100-elitism
NUM_POPULATION=40 #population size
MUTATION_RATE=0.50 #mutation probability
NUM_MUTATION=25



NUM_ITERATION=100
COUNTER_ITERATION=0

class OnePopul():
    def __init__(self,fitnes, distance, popul):
        self.popul=popul
        self.fitnes=fitnes
        self.distance=distance
       
        


class Population:
    def __init__(self,population, cities ): 
        self.cities=cities # tip Cities
        self.population=self.FillPopul(population)        
      

    #sortirace populaciju po tome ko ima najbolji fitnes
    def FillPopul(self,population): #and sort population
        res=[]
        for el in population:
            dist=self.Distance_cities(el)
            pop=OnePopul(self.Fitnes(dist),dist,el)
            res.append(pop)

        res.sort(key=lambda x: x.fitnes, reverse=True)  #sortira od manjeg ka vecem broju        
        return res


    #kreira matricu u kojoj pise razdaljina medju gradovima
    def Distance_cities(self,some_population): #some population je zapravo niz gradova
        i=0
        res=0
        while(i+1<len(some_population)):
            city1=some_population[i]
            city2=some_population[i+1]
            res+=self.cities.distance_table[city1-1][city2-1]
            i+=1

        res+=self.cities.distance_table[city2-1][0]
        return res


    def Fitnes(self,num):
        return float(1/num)
    


#Create 1st population
def MakePopulation(cities):
    popul=[]
    nameCities=list(range(2, len(citiesObj.cities)+1))
    for i in range(NUM_POPULATION):       
        np.random.shuffle(nameCities)
        popul.append([1]+nameCities)

    population=Population(popul,cities)
    return population

def Swap(popul, ind1,ind2):
    pom=popul[ind1]
    popul[ind1]=popul[ind2]
    popul[ind2]=pom


def Selection(): #function newGeneration
    return None


def Crossover2(popul1Orig, popul2Orig):  #GOTOVO 

    if(len(popul1Orig)!=52  or len(popul2Orig)!=52):
        print('')
        print('input je manje od 52') 

    popul1=copy.deepcopy(popul1Orig)
    popul2=copy.deepcopy(popul2Orig)
   
    popul1.remove(1)
    popul2.remove(1)
   
    ind1=np.random.randint(1,len(popul2))
    ind2=np.random.randint(1,len(popul2))
    if(ind2<ind1):
        p=ind1
        ind1=ind2
        ind2=p

    ind1=40
    ind2=51

    res=[1]
    partMedium=popul1[ind1:ind2+1]
    
    counter=0
    i=0   
    while (i<len(popul2)):       
        if(counter<ind1 or counter>=ind2):     
            if ((popul2[i] not in partMedium )and (popul2[i] not in res)):
                res.append(popul2[i])
                counter+=1
            i+=1               
        #else:
        if not (counter<ind1 or counter>=ind2):
            res+=popul1[ind1:ind2+1]
            counter=ind2

    if(len(res)!=52):
        print('')
        print("error len res",len(res))
    return res


def Mutation(popul):#GOTOVO    
    if(np.random.uniform(0,1)<MUTATION_RATE):
        ind1=math.floor(np.random.uniform(1,len(popul))) #da mi ne bi promenio da mi je prvi 1
        ind2=math.floor(np.random.uniform(1,len(popul)))
        Swap(popul,ind1,ind2)


        

def Mutaton_Inversal(popul):
    ind1=np.random.randint(1,52)
    ind2=np.random.randint(1,52)
    if(ind2<ind1):
        p=ind1
        ind1=ind2
        ind2=p
   
    
    seq=popul[ind1:ind2]
    if(np.random.uniform(0,1)<MUTATION_RATE): #MUTATION_RATE      
        seq.reverse()
    popul2=[]
    popul2=popul[:ind1]
    popul2+=seq
    popul2+=popul[ind2:]
    if(len(popul2)<52):
        print('less, Mutation')
    return popul2



def Replacement(oldPopulation,newPopulation): #adding old best, adding new best rest
    resPopul=[]
    saveNum=SAVE_BEST_PERCENT/100*len(oldPopulation.population)
    resPopul=oldPopulation.population[:int(saveNum)] #best from old
   
    for i in range(0,len(oldPopulation.population)-len(resPopul)):
        resPopul.append(newPopulation.population[i]) #best from new 
    
    resPopul.sort(key=lambda x: x.fitnes, reverse=True)
    return resPopul

#choosing 2 parrents(Selection) for Crossover and then doing Mutation
def NewGeneration(population,oldPopulation): 
    newPopulation=[]    

    for p in range(0,int(len(population))):
        bestesHalf=population[:int(len(population)/2)]  
        #worstHalf=population[int(len(population)/2):int(len(population))]      
       
        pop2=random.choice(bestesHalf)  
        pop1=random.choice(population)         
         
        newPopul=Crossover2(pop2.popul,pop1.popul)#iz prvog dela uzimam sredinu       
        aftMut=Mutaton_Inversal(newPopul)       

        newPopulation.append(aftMut)

    if(len(newPopul)!=52):
        print("ERR")
 
    return newPopulation
       



if __name__=="__main__": 
    print("_________________________________________________________")
   
    citiesObj= CreateCities() 
    population=MakePopulation(citiesObj ) 
# population:
#  lista gradova(koridinate grada+id, distance tabela izmedju svakog grada)
#  population (population- lista koja je poredjana po fitnesu i ima kojim redom obilazi koji grad)
    
    
    progress=[]

    oldPopulation=population
    #for i in range(0,NUM_ITERATION):
    counter=0
    start=time.time()
    end=time.time()
    while(population.population[0].distance>9000 or end-start<2*60): #bilo and izmedju 
        prevDistance=population.population[0].distance

 #main part
#population.population ima listu koja ima elemente distance, fitness,1 popul niz sa predlogom redoslda gradova
#ova lista je uredjena po distance ili fitnes nzm po cemu (od manje ka vece)        

        newPopulation=NewGeneration(population.population,oldPopulation)
            
#da poredjem novu populaciju po fitnesu 
        objPopulation=Population(newPopulation,population.cities)
#population-oldPopulation,objPopulation-new Population
       
        resPopulation=Replacement(population,objPopulation)
        population.population=resPopulation
       

        #for(el in resPopulation):
         #   if(len(el)<52)):
         #       print('Replacement nije ok vidi main')


        progress.append(population.population[0].distance)

        COUNTER_ITERATION+=1
        counter+=1
        end2=time.time()
        if(counter%500==0):
            print(counter,population.population[0].distance,", prev distance was: ",prevDistance, "time: ",(end2-start)/60)
           
        if(counter%3000==0 or population.population[0].distance<6500):
            plt.plot(progress)
            plt.ylabel('Distance')
            plt.xlabel('Generation')
            plt.title('Genetic Algorithm on TSP problem')
           # plt.figtext(.8, .8, "Num population = "+str(NUM_POPULATION)+'\nMutation percent='+str(MUTATION_RATE)+'\nprvi rod: random\ndrugi u grupu najboljih')
            #plt.suptitle('Population size = %.2i \n Mutation percent = %.2f, Max num mutation=%1f\n Crossover rate =%1f  ' % (NUM_POPULATION, MUTATION_RATE,NUM_MUTATION, 100-SAVE_BEST_PERCENT), fontsize=8)
            plt.figtext(.6,.7, 'Population size = %.2i \nMutation percent = %.2f \nCrossover rate =%1f  ' % (NUM_POPULATION, MUTATION_RATE, 100-SAVE_BEST_PERCENT)+
             '\n\nCurrent min:%3f '%(population.population[0].distance)+'\ntime: %.3f min' %((end2-start)/60),fontsize=7)
            #plt.figtext(.6,.7, 'time: '+str((end2-start)/60),fontsize=7)
        
            plt.show()
        

    print(counter,population.population[0].distance,", prev distance was: ",prevDistance)
    end=time.time()
    print("time: ",(end-start)/60)


    print("num popul: "+str(NUM_POPULATION)+" distance end: "+str(population.population[0].distance))
    print('order city: ',population.population[0])

   
   



    












