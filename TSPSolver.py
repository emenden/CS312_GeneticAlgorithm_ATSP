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
        start_time = time.time()

        greedy_solver = GreedySolver(self._cities)
        greedy_solver.greedy()
        initial_population = greedy_solver.get_greedy_solutions()

        # ensure initial population is even numbered
        if len(initial_population)%2 !=  0:
            print("odd initial greedy popn, adding 1 random tour")
            bssf_to_append = self.defaultRandomTour(start_time, time_allowance)
            while TSPSolution(bssf_to_append)==np.inf:
                bssf_to_append = self.defaultRandomTour(start_time, time_allowance)
            initial_population.append(bssf_to_append)

        # find the initial bssf
        self.set_initial_bssf_soln(initial_population)
        self.population = initial_population

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

    def branchAndBound( self, start_time, time_allowance=60.0 ):        # total time: O(n!), space: O(n^2)
        start_time = time.time()

        result = self.greedy(start_time, time_allowance=60.0) # time: O(n^2), space: O(n)
        # result = self.defaultRandomTour(start_time, time_allowance=60.0)
        bssfRoute = result.get("soln")
        BSSFnum = bssfRoute.costOfRoute()    # time: O(n), space: O(1)
        statesCreated = 0
        statesPruned = 0
        heap = MyHeap()
        count = 0   # don't count the initial greedy BSSF

        cities = self._scenario.getCities()     # time: O(1), space: O(n)
        self.numCities = len(cities)

        matrix = np.full((self.numCities, self.numCities), np.inf)     # time: O(n^2), space: O(n^2)
        lowerBound = 0
        route = []
        depth = 0
        currentCity = cities[0]

        # initialize distance matrix
        # set edge distances entering and leaving each city
        for current in cities:     # repeats n times, so time: O(n^2), space: O(1)
            for child in cities:     # repeats n times, so time: O(n), space: O(1)
                if current.getName() is not child.getName():  # time: O(1), space: O(1)
                    currToChildDist = current.costTo(child)
                    matrix[current.getIndex(), child.getIndex()] = currToChildDist

        # find Reduced Cost Matrix
        tuple = self.reducedCost(matrix, lowerBound)       # time: O(n^2), space: O(n^2)
        lowerBound = tuple[0]
        matrix = tuple[1]
        route.append(currentCity)

        # state: (depth, lowerBound, matrix, route)
        currentState = {'depth':depth, 'lowerBound':lowerBound, 'matrix':matrix, 'route':route}        # time: O(1), space: O(n^2)
        statesCreated += 1

        # add each child to the heap, unless its lowerBound is larger than the BSSF
        if lowerBound < BSSFnum:
            heap.push(currentState)     # time: O(n), space: O(1)

        # stop = 1
        # citiesToVisit = cities.copy()
        # citiesToVisit[currentCity.getIndex()] = None

        n = self.numCities
        while not heap.size() == 0 and time.time() <= (start_time + time_allowance):    # WORST CASE: repeats n! times (for each state in heap), so time: O(n!), space: O(n^2)
            currentState = heap.pop()       # time: O(logn), space: O(1)

            while currentState["lowerBound"] >= BSSFnum:       # WORST case: repated/prune n! times, so time: O(n!), space: O(1)
                currentState = heap.pop()       # time: O(logn), space: O(1)
                statesPruned += 1
                if currentState is None:
                    break
            if currentState is None:
                break
            currentCity = currentState["route"][-1]
            # citiesToVisit[currentCity.getIndex()] = None

            # find children of current city / "expand state"
            for dest in cities:       # repeats n times, so time: O(n^3), space: O(n^2)

                for visited in currentState["route"]:       # could repeat up to n times, so time: O(n), space: O(1)
                    # if the dest city has been visted in this route, set dest to None
                    if dest.getName() is visited.getName():       # time: O(1), space: O(1)
                        dest = None
                        break

                # if city has not been visited..
                if dest is not None:
                    # row = currentCity.getIndex()
                    # col = nextCity.getIndex()
                    destState = self.createState(currentState, dest)        # time: O(n^2), space: O(n^2)
                    statesCreated += 1

                    # if this state makes a full solution
                    if len(destState["route"]) == self.numCities:
                        # if this solution has a lower cost, replace BSSF
                        # if destState["lowerBound"] < BSSFnum:
                        if TSPSolution(destState["route"]).costOfRoute() < BSSFnum:
                            # if destState["route"][-1].costTo(cities[0]) < np.inf:
                            bssfRoute = TSPSolution(destState["route"])
                            BSSFnum = bssfRoute.costOfRoute()    # time: O(n), space: O(1)
                            count += 1

                    # if this state's LB is less than BSSF cost, add to heap
                    # else, prune this state (don't add to heap)
                    if destState["lowerBound"] < BSSFnum:
                        heap.push(destState)     # time: O(n), space: O(1)
                    else:
                        statesPruned += 1

        largestHeapSize = heap.getLargestSize()     # time: O(1), space: O(1)
        # statesCreated
        # statesPruned

        print("largest heap size:", largestHeapSize, "\nstates created:", statesCreated, "\nstates pruned:", statesPruned, "\n")


        results = {}
        # if not np.isinf(bssfRoute.costOfRoute()):
        results['cost'] = bssfRoute.costOfRoute()  # costOfBssf().ToString();   # time: O(n), space: O(1)
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssfRoute     # time: O(1), space: O(n)
        return results


    '''
    This method will run for 60 seconds, creating generations by calling 
    "survive_the_fittest()" which returns the new generation. 

    This method will create a "results" dictionary expected by the gui 
    and return that to the greedy() method which called it. 
    '''
    def fancy(self, start_time, time_allowance=60.0):
        self.timeallowance = time_allowance
        self.bssfChangeCount = 0
        self.sinceBSSFchanged = 0

        start_time = time.time()
        self.start_time = start_time
        self.time_allowance = time_allowance

        self.greedy(time.time(), time_allowance=60.0)
        population = self.population.copy()
        filteredPopulation = []
        for member in population:
            if member.costOfRoute() != np.inf:
                filteredPopulation.append(member)
        currentPopulation = filteredPopulation.copy()

        numCities = len(filteredPopulation[0].route)

        cycles = 0
        maxCycles = numCities//10
        if maxCycles < 5:
            maxCycles = 5

        # TODO will create generations until time is up, may want to regulate by num gens created
        while (time.time() - start_time) < time_allowance and cycles < maxCycles:
            if self.sinceBSSFchanged < 50:
                currentPopulation = self.survive_the_fittest(currentPopulation)
                self.sinceBSSFchanged += 1
            else:
                cycles += 1
                currentPopulation = filteredPopulation.copy()
                self.sinceBSSFchanged = 0


        # return the best solution found
        results = {}
        results['cost'] = self._bssf.costOfRoute() 
        results['time'] = time.time() - start_time
        results['count'] = self.bssfChangeCount    # TODO will need to change this to num updates to self._bssf?
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
        import operator
        remainingParents = population.copy()
        survivingPopulation = []

        # while there are still parents to combine
        while len(remainingParents) != 0:
            # get first random parent from remaining parents and delete from remainingParents list
            index1 = random.randint(0, len(remainingParents)-1)
            parent1 = self.Route(remainingParents[index1])
            del remainingParents[index1]

            # get second random parent from remaining parents and delete from remainingParents list
            index2 = random.randint(0, len(remainingParents)-1)
            parent2 = self.Route(remainingParents[index2])
            del remainingParents[index2]

            # combine: cross parents, then mutate the results to get the two children
            # cross1, cross2 = self.crossover(parent1, parent2)
            # child1, child2 = self.mutate(cross1, cross2)
            mut1, mut2 = self.mutate(parent1, parent2)
            child1, child2 = self.crossover(mut1, mut2)

            # find the two routes with the smallest cost (between both parents and both children)
            familyMembers = [parent1, parent2, child1, child2]
            familyMembers.sort(key=operator.attrgetter('cost'))
            survivor1 = familyMembers[0]
            survivor2 = familyMembers[1]
            i = 0

            if TSPSolution(survivor1.route).costOfRoute() < self._bssf.costOfRoute():
                cityList = survivor1.route
                self._bssf = TSPSolution(cityList.copy())
                self.bssfChangeCount += 1
                self.sinceBSSFchanged = 0

            # add the two smallest routes to the survivingPopulation list
            survivingPopulation.append(TSPSolution(survivor1.route))
            survivingPopulation.append(TSPSolution(survivor2.route))
        return survivingPopulation

    '''
    This method takes in two parent solutions which are TSPSolution objects.

    This method will take two parents from the population, create two children through
    crossing over. 

    This method returns two solutions created by crossing over the parents passed in.
    '''

    def crossover(self, parentA, parentB):
        child1 = self.makeChild(parentA, parentB)
        child2 = self.makeChild(parentB, parentA)
        stop1 = 0
        stop2 = 0
        while stop1 < 30 and child1.cost == np.inf and (time.time() - self.start_time) < self.time_allowance:  # or np.inf?
            child1 = self.makeChild(parentA, parentB)
            stop1 += 1
        while stop2 < 30 and child2.cost == np.inf and (time.time() - self.start_time) < self.time_allowance:
            child2 = self.makeChild(parentB, parentA)
            stop2 += 1
        return child1, child2

    def makeChild(self, parentA, parentB):
        from random import randint
        pLen = len(parentA.route)
        childpath = [-1] * pLen
        start = randint(0, pLen)
        stop = randint(0, pLen)
        while start == stop:
            stop = randint(0, pLen)
        if stop < start:
            temp = start
            start = stop
            stop = temp
        childpath = [-1] * pLen
        childpath[start:stop] = parentA.route[start:stop]
        p = 0
        for c in range(pLen):
            while parentB.route[p] in childpath:    # if city is already in childpath, then check next city
                p += 1
                if p >= len(parentB.route):
                    return self.Route(TSPSolution(childpath))
            # if childpath[c] != -1:  # if childpath[c] is not empty, then check next city in parentB
            #     continue
            if childpath[c] == -1:
                childpath[c] = parentB.route[p]     # set parentB city at p to childpath[c]
                p += 1
                if p >= len(parentB.route):
                    return self.Route(TSPSolution(childpath))
        return self.Route(TSPSolution(childpath.copy()))


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
        # implementation of mutation rate? -- mutation rate could increase
        # as itereations unchanged bssf increases, and in a high mutation
        # rate condition, we might implement greater change to the children
        costA = np.inf
        costB = np.inf
        # while costA == np.inf and (time.time() - self.start_time) < self.time_allowance:
        to_swap = random.sample(range(len(childA.route)), 2)
        childARoute = childA.route.copy()
        to_swap = random.sample(range(len(childB.route)), 2)
        childBRoute = childB.route.copy()


        for n in range(len(childARoute)//5):
            temp1 = childARoute[to_swap[0]]
            childARoute[to_swap[0]] = childARoute[to_swap[1]]
            childARoute[to_swap[1]] = temp1
        # muteA = self.Route(TSPSolution(childARoute))
        # costA = childA.getCost() #cost
        # while costB == np.inf and (time.time() - self.start_time) < self.time_allowance:

        # to_swap = random.sample(range(len(childB.route)), 2)
        # childBRoute = childB.route.copy()

            temp2 = childBRoute[to_swap[0]]
            childBRoute[to_swap[0]] = childBRoute[to_swap[1]]
            childBRoute[to_swap[1]] = temp2

        muteA = self.Route(TSPSolution(childARoute))
        muteB = self.Route(TSPSolution(childBRoute))
        return (muteA, muteB)


    class Route:
        def __init__(self, tspSolution):
            # self.route= route
            self.route = (tspSolution.route).copy()
            self.cost = TSPSolution(self.route).costOfRoute()
        def getRoute(self):
            return self.route
        def getCost(self):
            self.cost = TSPSolution(self.route).costOfRoute()
            return self.cost


class MyHeap:
    def __init__(self): #, initial=None, key=lambda x: x):
        self.heap = []
        self.largestSize = 0

    def getPriorityNum(self, state):        # time: O(1), space: O(1)
        # state: (depth, lowerBound, matrix, route)
        # priority number: lowerBound - depth
        if state['depth'] == 0:
            return state['lowerBound']
        return state['lowerBound']//state['depth']

    def push(self, state):        # total time: O(n), space: O(n^2)
        priorityNum = self.getPriorityNum(state)        # time: O(1), space: O(1)
        routeName = ""
        for city in state["route"]:        # time: O(n), space: O(1)
            routeName += city.getName()
        item = (priorityNum, routeName, state)  # time: O(1), space: O(n^2)
        # pushes item onto stack: (priorityNum, state)
        heapq.heappush(self.heap, item)     # time: O(logn), space: O(1)
        if len(self.heap)> self.largestSize:
            self.largestSize = len(self.heap)

    def pop(self):       # time: O(logn), space: O(n^2)
        # remove item from stack
        if len(self.heap)==0:
            return None
        item = heapq.heappop(self.heap)     # time: O(logn), space: O(n^2)
        # return only the state (not the priority number)
        return item[2]

    def size(self):       # time: O(1), space: O(1)
        return len(self.heap)

    def getLargestSize(self):       # time: O(1), space: O(1)
        return self.largestSize