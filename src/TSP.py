import numpy as np
import random
import time
import Utilis


def read_from_file(filename):
    # Read distance matrix from file.
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return distanceMatrix


class TSP:
    """ Parameters """
    def __init__(self, fitness, filename):
        self.alpha = 0.05             # Mutation probability
        self.lambdaa = 200           # Population size
        self.mu = self.lambdaa * 2    # Offspring size        WHY THE DOUBLE (COULD BE THE HALF?)
        self.k = 5                    # Tournament selection
        self.numIters = 10            # Maximum number of iterations
        self.objf = fitness                # Objective function

        self.distanceMatrix = read_from_file(filename)
        self.numCities = self.distanceMatrix.shape[0]         # Boundary of the domain, not intended to be changed

        # Initialize population
        self.population = np.zeros((self.lambdaa, self.numCities-1)).astype(int)
        for i in range(self.lambdaa):
            self.population[i, :] = self.random_cycle()

    """ The main evolutionary algorithm loop """
    def optimize(self):
        for i in range(self.numIters):
            # The evolutionary algorithm
            start = time.time()

            selected = self.selection(self.population, self.k)                  # selected = initial*2
            offspring = self.crossover(selected, self.k)                        # offspring = initial 
            joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), self.population))    # joinedPopulation = polutation + mutated children = lambdaa*2
            self.population = self.elimination(joinedPopulation, self.lambdaa)                         # population = joinedPopulation - eliminated = lambdaa

            itT = time.time() - start

            # Show progress
            # fvals = np.array([self.objf(path) for path in self.population])
            fvals = pop_fitness(self.population, self.distanceMatrix)
            meanObj = np.mean(fvals)
            bestObj = np.min(fvals)
            print(f'{itT: .2f}s:\t Mean fitness = {meanObj: .5f} \t Best fitness = {bestObj: .5f} \t pop shape = {tsp.population.shape}')
        print('Done')

    def selection_kTour(self, population, k):
        randIndices = random.choices(range(np.size(population,0)), k = k)
        best = np.argmin(pop_fitness(population[randIndices, :], self.distanceMatrix))
        return population[randIndices[best], :]

    """ Perform k-tournament selection to select pairs of parents. """
    def selection(self, population, k):        
        selected = np.zeros((self.mu, self.numCities - 1)).astype(int)
        for i in range(self.mu):
            selected[i, :] = self.selection_kTour(population, k)
        return selected

    """ Perform box crossover as in the slides. """
    def crossover(self, population, k):
        offspring = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
        for i in range(self.lambdaa):
            p1 = self.selection_kTour(population, k)
            p2 = self.selection_kTour(population, k) 
            subpath = Utilis.longest_common_subpath(p1, p2)
            restPath = np.setdiff1d(p1, subpath)
            np.random.shuffle(restPath)
            # print(subpath.shape)
            # print(restPath.shape)
            # print(offspring.shape)
            offspring[i, :] = np.append(subpath, restPath)
        return offspring

    """ Perform mutation, adding a random Gaussian perturbation. """
    def mutation(self, offspring, alpha):
        i = np.where(np.random.rand(np.size(offspring, 0)) <= alpha)[0]
        offspring[i, :] = self.random_cycle()
        return offspring

    """ Eliminate the unfit candidate solutions. """
    # TODO consider age-based elimination
    def elimination(self, population, lambdaa):
        fvals = pop_fitness(population, self.distanceMatrix)
        sortedFitness = np.argsort(fvals)
        return population[sortedFitness[0: lambdaa], :]     # TODO check: lambdaa - 1

    def random_cycle(self):
        path_len = self.distanceMatrix.shape[0]
        nodes = np.arange(path_len)
        frontier = [(0, [0])]
        expanded = set()

        while frontier:
            u, path = frontier.pop()
            if u == 0 and expanded:
                # found a cycle
                if len(path) == path_len + 1:
                    return np.array(path[1:-1])
            if u in expanded:
                continue
            expanded.add(u)

            # loop through the neighbours at a random order, to result to order in the frontier
            np.random.shuffle(nodes)
            for v in nodes:
                if (v != u) and (self.distanceMatrix[u][v] != np.inf):
                    # this is a neighbour
                    frontier.append((v, path + [v]))

        # in case it got to a dead end, rerun
        return self.random_cycle()

def pop_fitness(population, distanceMatrix):
    return np.array([fitness(path, distanceMatrix) for path in population])


""" Compute the objective function of a candidate"""
def fitness(path, distanceMatrix):
    sum = distanceMatrix[0, path[0]]
    for i in range(len(path)-2):
        if np.isinf(distanceMatrix[path[i]][path[i + 1]]):
            return np.inf
        else:
            sum += distanceMatrix[path[i]][path[i + 1]]
    sum += distanceMatrix[path[len(path) - 1]][0]  # cost from end of path back to begin of path
    return sum


tsp = TSP(fitness, "../data/tour50.csv")

# Testing selection
# pop = tsp.selection(tsp.population, 5)
# print(pop.shape)
# print(np.unique(pop, axis=0).shape)

# Testing mutation
# np.set_printoptions(threshold=np.inf)
# pop1 = tsp.population
# print(pop1)
# pop2  = tsp.mutation(tsp.population, tsp.alpha)
# print(pop2)

# Testing crossover
# pop1 = tsp.population
# pop2 = tsp.crossover(pop1, tsp.k)
# print(pop2)
# print(pop2.shape)

# Testing elimination
# pop2 = tsp.elimination(pop1, tsp.lambdaa)
# print(pop1.shape)
# print(pop2.shape)
# mean_fitness1 = sum(pop_fitness(pop1, tsp.distanceMatrix)) / len(pop1)
# print(mean_fitness1)
# pop2 = tsp.elimination(pop1, tsp.lambdaa)
# mean_fitness2 = sum(pop_fitness(pop2, tsp.distanceMatrix)) / len(pop2)
# print(mean_fitness2)

tsp.optimize()


