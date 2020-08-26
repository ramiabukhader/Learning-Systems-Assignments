import numpy as np
import pandas
import random as r

class ga:
    def cal_pop_fitness(inputs, pop):
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
       
        parents = []
        sort_fitness = fitness.copy()
        sort_fitness.sort()
        for i in range(num_parents):
            for x in fitness:
                if x == sort_fitness[i]:
                    parents.append(pop[fitness.index(x)])
                    break

        return parents

    def crossover(parents):
       
        offspring = []
        for i in range((len(parents))) :
            cross_parent=[]
            for x in range(len(parents[i])):
            
                if x < len(parents[i])/2:
                    cross_parent.append(parents[i][x])
                else:
                    for y in parents[(i+1)%len(parents)]:
                        if cross_parent.count(y) == 0:
                            cross_parent.append(y)
                            break
            for s in cross_parent:
                if cross_parent.count(s)>1:
                     print("error")                  
            offspring.append(cross_parent)          
        return offspring

    def mutation(offspring_crossover, num_mutations=1):
       
       
        # for i in offspring_crossover:
        #     for x in range(num_mutations):
        #         ran_loc = r.choices(i, k=2)
        #         tmp = i.index(ran_loc[1])
        #         i[i.index(ran_loc[0])] = ran_loc[1]
        #         i[tmp] = ran_loc[0]

        offspring_crossover_new=[]
        for i in offspring_crossover:
            index1=np.random.randint(1,52)
            index2=np.random.randint(1,52)
            if(index2<index1):
                indValue=index1
                index1=index2
                index2=indValue
            
            selected_part=i[index1:index2]
            if(np.random.uniform(0,1)<0.5):      
                selected_part.reverse()
            
            new_mut_child = i[:index1] + selected_part + i[index2:]
            offspring_crossover_new.append(new_mut_child)
        return offspring_crossover_new


def main():
  
    df = pandas.read_csv('berlin52.csv', header = None)
    inputs = np.array(df.iloc[:,:].values.tolist())
    
    # Inputs of the equation.
    equation_inputs = [i for i in range(53)][1:]

    # Number of the weights we are looking to optimize.
    num_weights = len(equation_inputs)

    sol_per_pop = 50
    num_parents_mating = 20

    # Defining the population size.
    pop_size = (sol_per_pop,num_weights) 
    
    new_population = []
    for i in range(sol_per_pop):
        r.shuffle(equation_inputs)
        new_population.append(equation_inputs.copy())
    
    best_outputs = []
    num_generations = 10000
    generation_history =[]
    generation_fitnness_history =[]
    for generation in range(num_generations):
        # Measuring the fitness of each chromosome in the population.
        fitness = ga.cal_pop_fitness(inputs, new_population)
        
        #best_outputs.append(np.max(np.sum(new_population*equation_inputs, axis=1)))
        best_outputs = fitness.copy()

        # The best result in the current iteration.
        print(f"Best result {generation} : ", np.min(best_outputs))
        generation_history.append(generation)
        generation_fitnness_history.append(np.min(best_outputs))
        
        if np.min(best_outputs) <8858:
            break
        
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
        
        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents)#,
                                        

        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=1)
        
        # Creating the new population based on the parents and offspring.
        new_population_copy = parents[:10]
        random_population_index = [i for i in range(sol_per_pop)][:]
        r.shuffle(random_population_index)
        for _index in range(20):
            new_population_copy.append(new_population[random_population_index[_index]])
        
        for i in offspring_mutation:
            new_population_copy.append(i)
        new_population = new_population_copy
        
    fitness = ga.cal_pop_fitness(inputs, new_population)
    
    best_match_idx = fitness.index(min(fitness))
    print("Best solution : ", new_population[best_match_idx])
    print("Best solution fitness : ", fitness[best_match_idx])


    import matplotlib.pyplot
    matplotlib.pyplot.plot(generation_history, generation_fitnness_history, label = "Traning")
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.show()

if __name__ == '__main__':
     main()