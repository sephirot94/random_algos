from collections import defaultdict
class CyclicDirGraph:

    # Time Complexity: O(V+E).
    # Time Complexity of this method is same as time complexity of DFS traversal which is O(V+E).
    # Space Complexity: O(V).
    # To store the visited and recursion stack O(V) space is needed.

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vtx = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def recursive_helper(self, v, visited, recStack):
        # Mark the current node as visited and add to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours if any neighbour is visited and in recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                if self.recursive_helper(neighbour, visited, recStack):
                    return True
            elif recStack[neighbour]:
                return True

        # The node needs to be poped from recursion stack before functions ends
        recStack[v] = False
        return False

    # Return true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * (self.vtx+1)
        recStack = [False] * (self.vtx+1)
        for node in range(self.vtx):
            if not visited[node]:
                if self.recursive_helper(node, visited, recStack):
                    return True
        return False

class CyclicUndirGraph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vtx = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def recursive_helper(self, v: int, visited: list, parent: int):
        # Mark the current node as visited and add to recursion stack
        visited[v] = True


        # Recur for all neighbours if any neighbour is visited and in recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                if self.recursive_helper(neighbour, visited, v):
                    return True
            elif neighbour != parent:
                return True

        return False

    # Return true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * (self.vtx+1)
        for node in range(self.vtx):
            if not visited[node]:
                if self.recursive_helper(node, visited, -1):
                    return True
        return False

class CyclicUndirGraphWithUnionFind:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A utility function to find the subset of an element i
    def find_parent(self, parent, i):
        if parent[i] == -1:
            return i
        if parent[i] != -1:
            return self.find_parent(parent, parent[i])

    # A utility function to do union of two subsets
    def union(self, parent, x, y):
        parent[x] = y

    # The main function to check whether a given graph
    # contains cycle or not
    def isCyclic(self):

        # Allocate memory for creating V subsets and
        # Initialize all subsets as single element sets
        parent = [-1] * (self.V)

        # Iterate through all edges of graph, find subset of both
        # vertices of every edge, if both subsets are same, then
        # there is cycle in graph.
        for i in self.graph:
            for j in self.graph[i]:
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent, x, y)
