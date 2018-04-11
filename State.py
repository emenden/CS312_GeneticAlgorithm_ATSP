'''
This class stores the information in the expansion tree of the TSP algorithm

space: O(n^2): this class holds an nxn matrix as well as a list of 
TSPClasses.City objects, max size n-1; these are the significant space contributors
'''
class State:

	def __init__(self, index, cost, lb, p, d, to_visit):
		self.leaving_index = index
		self.cost_matrix = cost
		self.lower_bound = lb
		self.parent = p
		self.depth = d
		self.score = lb/d
		self.unvisited_cities = to_visit

	def __lt__(self, other):
		return self.score < other.score

	def setPath(self, path):
		self.path = path

	def setDepth(self, depth):
		self.depth = depth

	def getScore(self):
		return self.score

	def get_leaving_index(self):
		return self.leaving_index

	def get_lower_bound(self):
		return self.lower_bound

	def get_cost(self):
		return self.cost_matrix

	def get_unvisited_cities(self):
		return self.unvisited_cities

	def get_parent(self):
		return self.parent

	def get_depth(self):
		return self.depth