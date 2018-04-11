#!/usr/bin/python3

import time
import numpy as np
from TSPClasses import *
import heapq
from State import State

class GreedySolver:
	'''
	This class implements a greedy algorithm for the TSP. It creates 
	a greedy tour starting at each unique city. If there is not a valid 
	greedy tour, it does not create a solution for that starting city.
	'''
	def __init__(self, cities):
		self._cities = cities
		self._greedy_solns = []


	def get_greedy_solutions(self):
		return self._greedy_solns


	'''
    time: this method calls O(n^2) functions with no loops or larger complexities 
    and so the time complexity is O(n^2)
    space: O(n^2) to hold the cost_matrix created which is nxn, the root state
    created is also O(n^2)
    '''
	def greedy(self):

		num_cities = len(self._cities)
		cost_matrix = np.zeros((num_cities, num_cities))

		self.form_cost_matrix(cost_matrix)  # time: O(n^2)

		lower_bound, cost = self.reduce_cost_matrix(cost_matrix)

		# loop through all cities as starting city
		for i in range(0, len(self._cities)):
			unvisited_cities = list(self._cities)
			del unvisited_cities[i]
			state = State(i, cost, lower_bound, None, 1, unvisited_cities)

			# create solution, append it to list of solutions
			solution = self.greedy_visit(state, i)
			if solution != None:
				self._greedy_solns.append(solution)


	'''
	time: O(n^2) because we are forming an nxn matrix, this assumes the City.costTo() 
	method takes insignificant time
	space: O(n^2) to hold the nxn array passed in as param
	'''
	def form_cost_matrix(self, cost_matrix):
		i = 0
		j = 0
		for row_city in self._cities:   # O(n)
			for col_city in self._cities:   # O(n)
				if row_city.getName() == col_city.getName():
					cost_matrix[i, j] = np.inf
				else:
					cost_matrix[i, j] = row_city.costTo(col_city)   # insignificant time
				j = j+1
			i = i+1
			j = 0


	'''
	time: this method will run in O(n^2): it is only looking at n nodes because 
	it only expands the nearest city therefore, this part of the algorithm is O(n),
	however, there are some calls to O(n^2) functions to reduce the cost matrixes
    space: this algorithm uses O(n^3) space to hold the O(n^2) state objects, of 
    which there are n
    '''
	def greedy_visit(self, state, start_city_index):
		all_cities_visited = False
		while(not all_cities_visited):
			leaving_index = state.get_leaving_index()
			arriving_index = self.find_nearest_city(state, start_city_index)
			if arriving_index == None:	# nearest city is inf cost away
				return None	
			
			cost_to_travel = state.get_cost()[leaving_index, arriving_index]
			parent_unvisited_cities = state.get_unvisited_cities()

			child_depth = state.get_depth()+1
			child_cost = self.visit_city(state.get_cost(), leaving_index, arriving_index)  # O(n^2) function call
			lower_bound_addon, reduced_cost = self.reduce_cost_matrix(child_cost)   # O(n^2) function call
			lower_bound = state.get_lower_bound() + lower_bound_addon + cost_to_travel

			# update the list of unvisited cities for the child
			child_unvisited_cities = list(parent_unvisited_cities)  # O(n-1) max space
			arriving_city = self._cities[arriving_index]

			i=0
			for city in child_unvisited_cities: # O(n) time complexity
				if city.getName() == arriving_city.getName():
					del child_unvisited_cities[i]
					break
				i=i+1

			child_state = State(arriving_index, reduced_cost, lower_bound, state, child_depth, child_unvisited_cities)
            
			# check if solution, if valid solution return it, else ret null
			if self.check_if_solution(child_state):   # O(1) call time complexity
				all_cities_visited = True
                
				bssf = self.form_potential_bssf(child_state)
				if bssf.costOfRoute() < np.inf:
					return bssf
				else:
					return None
			# update where we are in the route before repeating loop
			state = child_state


	'''
	time: O(n^2) to find the min values in each row (I believe np.argmin is O(n^2))
	space: O(n^2) to hold the state object passed in as param
	'''
	def find_nearest_city(self, state, start_city_index):
		leaving_index = state.get_leaving_index()
		mins_rows = np.argmin(state.get_cost(), axis=1)
		nearest_city_index = mins_rows[leaving_index]
		if nearest_city_index == leaving_index or (nearest_city_index == start_city_index and len(state.get_unvisited_cities()) > 0): # not finished with tour, find other min
			row_indexes_sorted = np.argsort(state.get_cost()[leaving_index])
			nearest_city_index = row_indexes_sorted[1]
		if state.get_cost()[leaving_index][nearest_city_index] == np.inf:
			return None
		return nearest_city_index


	'''
	time: O(n^2) because of the two calls to O(n^2) functions, row_reduce()
	and col_reduce()
	space: O(n^2) to hold the nxn matrixes created and the cost matrix passed in
	'''
	def reduce_cost_matrix(self, cost):
		row_reduction, row_reduced_cost = self.row_reduce(cost)
		col_reduction, col_reduced_cost = self.col_reduce(row_reduced_cost)
		lower_bound_addon = row_reduction + col_reduction
		return lower_bound_addon, col_reduced_cost


	'''
	time: O(n^2): I believe np.amin would take at most O(n^2) time because it might need to 
	look at each element before a min is found for each row, and subtracting the mins 
    element-wise will be O(n^2)
    space: O(n^2) to hold the cost matrix, and O(n^2) to hold the row-reduced cost matrix
    O(n) to hold the mins of each row
	'''
	def row_reduce(self, cost_matrix):
		row_mins = np.amin(cost_matrix, axis=1)
		row_mins[row_mins == np.inf] = 0
		B = cost_matrix - row_mins.reshape(-1, 1)
		sum_row_reduction = row_mins.sum()
		return sum_row_reduction, B


	'''
    time: O(n^2): I believe np.amin would take at most O(n^2) time because it might need to 
    look at each element before a min is found for each column, and subtracting the mins 
    element-wise will be O(n^2)
    space: O(n^2): to hold the cost matrix, and O(n^2) to hold the column-reduced cost matrix
    O(n) to hold the mins of each column
    '''
	def col_reduce(self, cost_matrix):
		col_mins = np.amin(cost_matrix, axis=0)
		col_mins[col_mins == np.inf] = 0
		C = cost_matrix - col_mins.reshape(1,-1)
		sum_col_reduction = col_mins.sum()
		return sum_col_reduction, C


	'''
    time: O(n^2) for copying the array, O(n) for updating the cost matrix because 
    I update the rows/columns with only 1 iterator
    space: O(n^2) for hold the array passed in as param, O(n^2) to hold copy of cost
    matrix passed in
    '''
	def visit_city(self, cost_matrix, leaving_index, arriving_index):
		cost = cost_matrix.copy()
		for i in range(0, len(self._cities)):
			cost[i, arriving_index] = np.inf
			cost[leaving_index, i] = np.inf
		cost[arriving_index, leaving_index] = np.inf
		return cost


	'''
    time: O(1) because simply looking up value for State param passed in
    space: O(n^2) to hold State object passe in
    '''
	def check_if_solution(self, child_state):
		if len(child_state.get_unvisited_cities()) == 0:
			return True
		else:
			return False


	def form_potential_bssf(self, child_state):
		parent = child_state.get_parent()
		route = [self._cities[child_state.get_leaving_index()]]
		while parent is not None:
			route.insert(0, self._cities[parent.get_leaving_index()])
			parent = parent.get_parent()
		bssf = TSPSolution(route)
		return bssf
