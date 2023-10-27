import numpy as np
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def read_from_file(filename):
	# Read distance matrix from file.
	file = open(filename)
	distanceMatrix = np.loadtxt(file, delimiter=",")
	file.close()
	return distanceMatrix


class TSP:
	""" Parameters """
	def __init__(self, fitness, filename):
		self.alpha = 0.05     		# Mutation probability
		self.lambdaa = 200   		# Population size
		self.mu = self.lambdaa * 2	# Offspring size		WHY THE DOUBLE (COULD BE THE HALF?)
		self.k = 2        			# Tournament selection
		self.numIters = 50			# Maximum number of iterations
		self.objf = fitness				# Objective function

		self.distanceMatrix = read_from_file(filename)
		self.numCities = self.distanceMatrix.shape[0]     	# Boundary of the domain, not intended to be changed.

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
			joinedPopulation = np.vstack((self.mutation(offspring, self.alpha), population))
			population = self.elimination(joinedPopulation, self.lambdaa)

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
	def selection(self, population, k):		# CHECK MU
		selected = np.zeros((self.mu, self.numCities))
		for i in range( self.mu ):
			selected[i, :] = self.selection_kTour(population, k)
		return selected


	""" Perform box crossover as in the slides. """
	def crossover(self, selected):
		weights = 3*np.random.rand(self.lambdaa,2) - 1
		offspring = np.zeros((self.lambdaa, 2))
		lc = lambda x, y, w: np.clip(x + w * (y-x), 0, self.numCities)
		for ii, _ in enumerate(offspring):
			offspring[ii,:] = lc(selected[2*ii, :], selected[2*ii+1, :], weights[ii, :])
		return offspring
	
	def crossover(self, population, k):

		p1 = self.selection_kTour(population, k)
		p2 = self.selection_kTour(population, k)
		longest_common_subpath(p1,p2)


		return



	""" Perform mutation, adding a random Gaussian perturbation. """
	def mutation(self, offspring, alpha):
		ii = np.where(np.random.rand(np.size(offspring,0)) <= alpha)
		offspring[ii,:] = offspring[ii,:] + 10*np.random.randn(np.size(ii),2)
		offspring[ii,0] = np.clip(offspring[ii,0], 0, self.numCities)
		offspring[ii,1] = np.clip(offspring[ii,1], 0, self.numCities)
		return offspring

	""" Eliminate the unfit candidate solutions. """
	def elimination(self, joinedPopulation, keep):
		fvals = self.objf(joinedPopulation)
		perm = np.argsort(fvals)
		survivors = joinedPopulation[perm[0:keep-1],:]
		return survivors


def pop_fitness(population):
	return np.array([fitness(path) for path in population])

""" Compute the objective function of a candidate"""
def fitness(path, distance_matrix):
	sum = distance_matrix[path[-1]][0] + distance_matrix[0][path[0]]		# index zero at the beginning is fixed

	for i in range(len(path)-1):
		sum = sum + distance_matrix[path[i]][path[i+1]] 

	return sum


tsp = TSP(fitness, filename)
tsp.optimize()
