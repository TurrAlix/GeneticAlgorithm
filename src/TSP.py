import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
        self.k = 2                    # Tournament selection
        self.numIters = 50            # Maximum number of iterations
        self.objf = fitness                # Objective function

        self.distanceMatrix = read_from_file(filename)
        self.numCities = self.distanceMatrix.shape[0]         # Boundary of the domain, not intended to be changed.

    """ The main evolutionary algorithm loop """
    def optimize(self):

        # Initialize population
        population = np.vstack([np.arange(1, self.numCities)] * self.lambdaa)
        for i in range(self.lambdaa):
            np.random.shuffle(population[i])

        for i in range(self.numIters):
            # The evolutionary algorithm
            start = time.time()

            selected = self.selection(population, self.k)
            offspring = self.crossover(selected)
            joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))    # joinedPopulation = polutation + children + mutated
            population = self.elimination(joinedPopulation, self.lambdaa)                         # population = joinedPopulation - eliminated

            itT = time.time() - start

            # Show progress
            fvals = np.array([self.objf(path) for path in population])
            meanObj = np.mean(fvals)
            bestObj = np.min(fvals)
            print(f'{itT: .2f}s:\t Mean fitness = {meanObj: .5f} \t Best fitness = {bestObj: .5f}')
        print('Done')

    def selection_kTour(population, k):
        randIndices = random.choices(range(np.size(population,0)), k = k)
        best = np.argmin(pop_fitness(population[randIndices, :]))
        return population[best, :]

    """ Perform k-tournament selection to select pairs of parents. """
    def selection(self, population, k):        # CHECK MU
        selected = np.zeros((self.mu, self.numCities))
        for i in range(self.mu):
            selected[i, :] = self.selection_kTour(population, k)
        return selected

    """ Perform box crossover as in the slides. """
    def crossover(self, population, k):
        offspring = np.zeros((self.lambdaa, 2))
        for i in range(self.lambdaa):
            p1 = self.selection_kTour(population, k)
            p2 = self.selection_kTour(population, k)
            subpath = Utilis.longest_common_subpath(p1, p2)
            restPath = np.setdiff1d(p1, subpath)
            np.random.shuffle(restPath)
            offspring[i, :] = np.append(subpath, restPath)
        return offspring

    """ Perform mutation, adding a random Gaussian perturbation. """
    def mutation(self, offspring, alpha):
        ii = np.where(np.random.rand(np.size(offspring, 0)) <= alpha)
        offspring[ii, :] = offspring[ii, :] + 10*np.random.randn(np.size(ii),2)
        offspring[ii, 0] = np.clip(offspring[ii, 0], 0, self.numCities)
        offspring[ii, 1] = np.clip(offspring[ii, 1], 0, self.numCities)
        return offspring

    def mutation(self, offspring, alpha):
        i = np.where(np.random.rand(np.size(offspring, 0)) <= alpha)
        offspring[i,:] = np.random.suffle(offspring[i,:])
        return offspring

    """ Eliminate the unfit candidate solutions. """
    def elimination(self, joinedPopulation, keep):
        fvals = self.objf(joinedPopulation)
        perm = np.argsort(fvals)
        survivors = joinedPopulation[perm[0:keep-1],:]
        return survivors

    # TODO consider age-based elimination
    def elimination(self, population, k):
        fvals = pop_fitness(population)
        sortedFitness = np.argsort(fvals)
        return population[sortedFitness[0:k-1], :]

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
                    return np.array(path)
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


def pop_fitness(population):
    return np.array([fitness(path) for path in population])


""" Compute the objective function of a candidate"""
def fitness(path, distance_matrix):
    sum = 0
    for i in range(len(path)-2):
        if np.isinf(distance_matrix[path[i]][path[i + 1]]):
            return np.inf
        else:
            sum += distance_matrix[path[i]][path[i + 1]]
    sum += distance_matrix[path[len(path) - 1], path[0]]  # cost from end of path back to begin of path
    return sum


tsp = TSP(fitness, filename)
tsp.optimize()
