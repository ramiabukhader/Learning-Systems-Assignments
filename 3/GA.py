import numpy as np
import pandas
import random as r

class ga:
    def cal_pop_fitness(inputs, pop):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function calulates the sum of products between each input and its corresponding weight.
        fitness = []
        for i in pop:
            pop_sum = 0
            for j in range(len(i)):
                x1 = inputs[i[j]-1][1]
                y1 = inputs[i[j]-1][2]
                x2 = inputs[i[(j+1)%52]-1][1]
                y2 = inputs[i[(j+1)%52]-1][2]
                pop_sum += ((x2-x1)**2 + (y2-y1)**2)**(1/2)
            fitness.append(pop_sum)
        return fitness

    def select_mating_pool(pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
      
        #choosing fathers

        parents = []
        sort_fitness = fitness.copy()
        sort_fitness.sort()
        for i in range(num_parents):
            for x in fitness:
                if x == sort_fitness[i]:
                    parents.append(pop[fitness.index(x)])
                    break

        return parents

    def crossover(parents):#, offspring_size):
        #offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        #crossover_point = np.uint8(offspring_size[1]/2)

       
        offspring = []
        for i in range(len(parents)) :
            cross_parent=[]
            for x in range(len(parents[i])):
            
                if x < len(parents[i])/2:
                    cross_parent.append(parents[i][x])
                else:
                    for y in parents[(i+1)%len(parents)]:
                        if cross_parent.count(y) == 0:
                            cross_parent.append(y)
                            break

            offspring.append(cross_parent)
       
                    
        return offspring

    def mutation(offspring_crossover, num_mutations=0.50):
        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
       
        for i in offspring_crossover:
            for x in range(num_mutations):
                ran_loc = r.choices(i, k=2)
                i[i.index(ran_loc[0])] = ran_loc[1]
                i[i.index(ran_loc[1])] = ran_loc[0]
                
        return offspring_crossover

def main():
  
    df = pandas.read_csv('berlin52.csv', header = None)
    inputs = np.array(df.iloc[:,:].values.tolist())
    #inputs = preprocessing.normalize(inputs)


    # Inputs of the equation.
    equation_inputs = [i for i in range(53)][1:]#[4,-2,3.5,5,-11,-4.7]

    # Number of the weights we are looking to optimize.
    num_weights = len(equation_inputs)

    """
    Genetic algorithm parameters:
        Mating pool size
        Population size
    """
    sol_per_pop = 40
    num_parents_mating = 25

    # Defining the population size.
    pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    #new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
    new_population = []
    for i in range(sol_per_pop):
        r.shuffle(equation_inputs)
        new_population.append(equation_inputs.copy())
    print(new_population)


    best_outputs = []
    num_generations = 100
    for generation in range(num_generations):
        #print("Generation : ", generation)
        #print(generation)
        # Measuring the fitness of each chromosome in the population.
        fitness = ga.cal_pop_fitness(inputs, new_population)
        #print("Fitness", len(fitness))
        #print(fitness)

        #best_outputs.append(np.max(np.sum(new_population*equation_inputs, axis=1)))
        best_outputs = fitness.copy()

        # The best result in the current iteration.
        #print("Best result : ", np.max(np.sum(new_population*equation_inputs, axis=1)))
        print("Best result : ", np.min(best_outputs))
        
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
        #print("Parents", len(parents))
        #print(parents)

        #parents = np.array(parents)
        #print(parents.shape[0])

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents)#,
                                        #offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        #print("Crossover", len(offspring_crossover))
        #print(offspring_crossover)

        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
        #print("Mutation", len(offspring_mutation))
        #print(offspring_mutation)

        # Creating the new population based on the parents and offspring.
        #new_population[0:parents.shape[0], :] = parents
        #new_population[parents.shape[0]:, :] = offspring_mutation
        new_population = parents
        #new_population.append(offspring_mutation[:,:])
        for i in offspring_mutation:
            new_population.append(i)
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = ga.cal_pop_fitness(inputs, new_population)
    # Then return the index of that solution corresponding to the best fitness.
    #best_match_idx = np.where(fitness == np.min(fitness))
    best_match_idx = fitness.index(min(fitness))
    print("Best solution : ", new_population[best_match_idx])
    print("Best solution fitness : ", fitness[best_match_idx])


    #import matplotlib.pyplot
   # matplotlib.pyplot.plot(best_outputs)
    #matplotlib.pyplot.xlabel("Iteration")
   # matplotlib.pyplot.ylabel("Fitness")
   # matplotlib.pyplot.show()

if __name__ == '__main__':
     main()