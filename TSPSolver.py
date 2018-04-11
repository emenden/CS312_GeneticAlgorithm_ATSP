#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
from GreedySolver import GreedySolver


class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None
        self._bssf = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario
        self._cities = scenario.getCities()


    def defaultRandomTour( self, start_time, time_allowance=60.0 ):
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []

            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )

            bssf = TSPSolution(route)
            count += 1

            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
                return bssf


    '''
    This method is called by GUI. It will create the initial set of greedy solutions
    which become the initial population. It will make sure the initial population is 
    even size, creating a random tour if odd size. 

    This method will call genetic_tsp() which will return the "results" dictionary 
    needed by the GUI. 
    '''
    def greedy( self, start_time, time_allowance=60.0 ):
        greedy_solver = GreedySolver(self._cities)
        greedy_solver.greedy()
        initial_population = greedy_solver.get_greedy_solutions()

        # ensure initial population is even numbered
        if len(initial_population)%2 !=  0:
            print("odd initial greedy popn, adding 1 random tour")
            bssf_to_append = self.defaultRandomTour(start_time, time_allowance)
            initial_population.append(bssf_to_append)

        # find the initial bssf
        self.set_initial_bssf_soln(initial_population)

        results = {}
        results['cost'] = self._bssf.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = 0
        results['soln'] = self._bssf
        return results

    def set_initial_bssf_soln(self, initial_solutions):
        min_solution = initial_solutions[0]
        for solution in initial_solutions:
            if solution.costOfRoute() < min_solution.costOfRoute():
                min_solution = solution
        self._bssf = min_solution


    '''
    This method will run for 60 seconds, creating generations by calling 
    "survive_the_fittest()" which returns the new generation. 

    This method will create a "results" dictionary expected by the gui 
    and return that to the greedy() method which called it. 
    '''
    def fancy(self, population, time_allowance=60.0):
        self.greedy(time.time(), time_allowance=60.0)

        start_time = time.time()
        # TODO will create generations until time is up, may want to regulate by num gens created
        while (time.time() - start_time) < time_allowance:
            population = self.survive_the_fittest(population)

        # return the best solution found
        results = {}
        results['cost'] = self._bssf.costOfRoute() 
        results['time'] = time.time() - start_time
        results['count'] = 0    # TODO will need to change this to num updates to self._bssf?
        results['soln'] = self._bssf

        return results


    '''
    This method takes as param the whole population which is a list<TSPSolution>. 

    It will call crossover on pairs of solutions in the population list. 
    It will call mutate on the children created by crossover().
    It will choose the two best solutions of the { parents , children } set (size 4)
    to survive to next generation. It will add survivors to a list<TSPSolution> to return.

    This method will also update self._bssf if better solution is found.

    This method will return list<TSPSolution> which will become the population
    for the next generation.
    '''
    def survive_the_fittest(self, population):
        pass


    '''
    This method takes in two parent solutions which are TSPSolution objects.

    This method will take two parents from the population, create two children through
    crossing over. 

    This method returns two solutions created by crossing over the parents passed in.
    '''
    def crossover(self, parentA, parentB):
        pass


    '''
    This method will take the two child solutions created by crossover() which are 
    TSPSolution objects.

    This method will create a valid mutation of each solution passed in as a parameter. A 
    "valid" mutation is one that has cost < np.inf. This method will continue to find 
    mutations of a given child solution until a valid mutation is found. 

    This method returns two solutions which are valid mutations of the solutions passed
    in as params.
    '''
    def mutate(self, childA, childB):
        pass
