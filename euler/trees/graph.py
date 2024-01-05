import sys
from collections import defaultdict
from heap.heapster import CustomHeap

class FindConnectedComponents:

    def __init__(self, V: int):
        self.vertices = V
        self.graph = [[] * self.V]

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs(self, arr: list, v: int, visited: int) -> list:
        """
        Returns an array of the components from DFS traversal.
        :param arr: array to traverse
        :param v: current vertex
        :param visited: visited array tracking previously visited nodes
        :return: list containing connected components (neighbors)
        """
        visited[v] = True  # mark node as visited
        arr.append(v)  # store the visited node to track connected nodes
        for i in self.graph[v]:  # iterate through adjacent nodes (neighbors)
            if not visited[i]:  # If neighbor has not been visited
                arr = self.dfs(arr, i, visited)

        return arr

    def find_connected_components(self) -> list:
        """
        Returns a list containing the connected components in the graph
        :return: list cointaining the sets of connected components (or connected nodes)
        """
        visited = [False for i in range(self.V)]
        connected = []
        for v in range(self.V):
            if not visited[v]:
                arr = []
                connected.append(self.dfs(arr, v, visited))

        return connected


class CustomNodeIslands:
    def __init__(self, id: str, val: int):
        self.id = id
        self.val = val


class CustomGraph:  # Used in Number of islands

    def __init__(self, V: int):
        self.vertices = V
        self.adj = defaultdict(list)

    def add_edge(self, u: CustomNodeIslands, v: CustomNodeIslands):
        """
        Adds edge to graph
        """
        self.adj[u.id].append(v)
        self.adj[v.id].append(u)

    def check_boundary(self, i, j, visited, mtx):
        return i < len(mtx) and j < len(mtx[i]) and mtx[i][j] and not visited[i][j]

    def convert_mtx_to_graph(self, mtx: list):
        """
        Converts given matrix to a graph
        """
        if not mtx:
            return None
        visited = [[False for col in mtx] for row in mtx]

        self.vertices = len(mtx)

        for i in range(len(mtx)):
            for j in range(len(mtx[i])):
                visited[i][j] = True
                node = CustomNodeIslands(str(i) + str(j), mtx[i][j])
                if self.check_boundary(i+1, j, visited, mtx):
                    neighbor = CustomNodeIslands(str(i+1) + str(j), mtx[i+1][j])
                    self.add_edge(node, neighbor)
                    visited[i+1][j]
                if self.check_boundary(i-1, j, visited, mtx):
                    neighbor = CustomNodeIslands(str(i - 1) + str(j), mtx[i - 1][j])
                    self.add_edge(node, neighbor)
                    visited[i - 1][j]
                if self.check_boundary(i, j+1, visited, mtx):
                    neighbor = CustomNodeIslands(str(i) + str(j+1), mtx[i][j+1])
                    self.add_edge(node, neighbor)
                    visited[i][j+1]
                if self.check_boundary(i, j-1, visited, mtx):
                    neighbor = CustomNodeIslands(str(i) + str(j-1), mtx[i][j-1])
                    self.add_edge(node, neighbor)
                    visited[i][j-1]


class CheckStronglyConnected:
    """Implementation of Kosaraju's algorithm for strongly connected components"""

    def __init__(self, V: int):
        self.vertices = V  # number of vertices
        self.graph = defaultdict(list)

    def add_edge(self, u: int, v: int):
        self.graph[u].append(v)

    def dfs(self, vertex: int, visited: list):
        visited[vertex] = True  # set current vertex as visited
        for i in self.graph:  # traverse graph
            if not visited[i]:  # if not visited
                self.dfs(i, visited)  # visit it

    def dfs_scc(self, v: int, visited: list, scc_arr: list) -> list:
        """Returns array with strongly connected components"""
        visited[v] = True
        scc_arr.append(v)
        for i in self.graph[v]:  # traverse all neighbors
            if not visited[v]:  # If not visited
                self.dfs_scc(i, visited, scc_arr)

        return scc_arr


    def transpose(self):
        """
        Returns transposed (or reversed) graph
        """
        transposed = CheckStronglyConnected(self.vertices)
        for i in self.graph:
            for j in self.graph[i]:
                transposed.add_edge(j, i)  # traverse graph and reverse it in g

        return transposed

    def fill_order(self, v, visited, stack):
        visited[v] = True  # mark node as visited
        for i in self.graph[v]:
            if not visited[i]:
                self.fill_order(i, visited, stack)

        stack.append(v)

    def check_strongly_connected(self) -> bool:
        """
        Checks whether graph is strongly connected
        """
        # Time complexity is O(V+E)
        # Step 1: Mark all vertices as not visited
        visited = [False for i in range(self.vertices)]

        #Step 2: traverse graph with DFS starting from first vertex
        self.dfs(0, visited)
        for i in visited:  # Check all nodes have been visited, otherwise not strongly connected
            if not i:
                return False

        # Step 3: Create reversed graph
        gr = self.transpose()

        # Step 4: Mark all vertices as not visited (for second DFS)
        visited = [False for i in range(self.vertices)]

        # Step 5: Do DFS for reversed graph, starting from same vertex as before
        gr.dfs(0, visited)

        for i in visited:  # Check all nodes have been visited, otherwise not strongly connected
            if not i:
                return False

        return True

    def find_all_strongly_connected(self):
        scc = []
        stack = []
        visited = [False for i in range(self.vertices)]
        for i in range(self.vertices):  # DFS traversal appending to stack
            self.fill_order(i, visited, stack)

        reversed = self.transpose()  # reverse graph

        visited = [False for i in range(self.vertices)]  # mark all vertices as not visited for second DFS
        while stack:  # process in order defined by stack (get SCC)
            vertex = stack.pop()
            if not visited[vertex]:
                scc_arr = []
                reversed.dfs_scc(vertex, visited, scc_arr)
                scc.append(scc_arr)

        return scc


class GraphAdjMatrix:
    # A simple representation of graph using Adjacency Matrix
    def __init__(self, numvertex):
        self.adjMatrix = [[-1] * numvertex for x in range(numvertex)]
        self.numvertex = numvertex
        self.vertices = {}
        self.verticeslist = [0] * numvertex

    def set_vertex(self, vtx, id):
        if 0 <= vtx <= self.numvertex:
            self.vertices[id] = vtx
            self.verticeslist[vtx] = id

    def set_edge(self, frm, to, cost=0):
        frm = self.vertices[frm]
        to = self.vertices[to]
        self.adjMatrix[frm][to] = cost
        # for directed graph do not add this
        self.adjMatrix[to][frm] = cost

    def get_vertex(self):
        return self.verticeslist

    def get_edges(self):
        edges = []
        for i in range(self.numvertex):
            for j in range(self.numvertex):
                if (self.adjMatrix[i][j] != -1):
                    edges.append((self.verticeslist[i], self.verticeslist[j], self.adjMatrix[i][j]))
        return edges

    def get_matrix(self):
        return self.adjMatrix


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
        self.graph[v].append(u)

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


class KruskalMinSpanningTree:
    """
    This class get's the MST of a tree using kruskal's algorithm. Greedy algorithm found in
    https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
    Minimum Spanning Tree (MST) problem: Given connected graph G with positive edge weights,
    find a min weight set of edges that connects all of the vertices.
    """

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self):

        # Time Complexity: O(ElogE) or O(ElogV). Sorting of edges takes O(ELogE) time.
        # After sorting, we iterate through all edges and apply the find-union algorithm.
        # The find and union operations can take at most O(LogV) time. So overall complexity is O(ELogE + ELogV) time.
        # The value of E can be at most O(V2), so O(LogV) is O(LogE) the same.
        # Therefore, the overall time complexity is O(ElogE) or O(ElogV)

        result = []  # This will store the resultant MST

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't
            #  cause cycle, include it in result
            #  and increment the indexof result
            # for next edge
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0
        print("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree", minimumCost)


class PrimsMinSpanningTree:
    """
    The idea behind Prim’s algorithm is simple, a spanning tree means all vertices must be connected.
    So the two disjoint subsets (discussed above) of vertices must be connected to make a Spanning Tree.
    And they must be connected with the minimum weight edge to make it a Minimum Spanning Tree. Algorithm:
    1) Create a set mstSet that keeps track of vertices already included in MST.
    2) Assign a key value to all vertices in the input graph. Initialize all key values as INFINITE. 
    Assign key value as 0 for the first vertex so that it is picked first.
    3) While mstSet doesn’t include all vertices
        ….a) Pick a vertex u which is not there in mstSet and has minimum key value.
        ….b) Include u to mstSet.
        ….c) Update key value of all adjacent vertices of u. To update the key values, iterate through all adjacent 
        vertices. For every adjacent vertex v, if weight of edge u-v is less than the previous key value of v, update
        the key value as weight of u-v
    The idea of using key values is to pick the minimum weight edge from cut. The key values are used only for vertices
    which are not yet included in MST, the key value for these vertices indicate the minimum weight edges connecting 
    them to the set of vertices included in MST. 
    """

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent):
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initilaize min value
        min = sys.maxint

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxint] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.printMST(parent)


class PrimsMSTAdjList:
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)

    def addEdge(self, src, dest, weight):
        """
        Adds an edge to an undirected graph
        """
        # Add an edge from src to dest.  A new node is
        # added to the adjacency list of src. The node
        # is added at the beginning. The first element of
        # the node has the destination and the second
        # elements has the weight
        newNode = [dest, weight]
        self.graph[src].insert(0, newNode)

        # Since graph is undirected, add an edge from
        # dest to src also
        newNode = [src, weight]
        self.graph[dest].insert(0, newNode)

    def PrimMST(self):
        """
        The main function that prints the Minimum Spanning Tree(MST) using the Prim's Algorithm
        """
        # O(ELogV)
        # Get the number of vertices in graph
        V = self.V

        # key values used to pick minimum weight edge in cut
        key = []

        # List to store contructed MST
        parent = []

        # minHeap represents set E
        minHeap = CustomHeap()

        # Initialize min heap with all vertices. Key values of all
        # vertices (except the 0th vertex) is is initially infinite
        for v in range(V):
            parent.append(-1)
            key.append(sys.maxint)
            minHeap.array.append(minHeap.newMinHeapNode(v, key[v]))
            minHeap.pos.append(v)

        # Make key value of 0th vertex as 0 so
        # that it is extracted first
        minHeap.pos[0] = 0
        key[0] = 0
        minHeap.decreaseKey(0, key[0])

        # Initially size of min heap is equal to V
        minHeap.size = V;

        # In the following loop, min heap contains all nodes
        # not yet added in the MST.
        while minHeap.isEmpty() == False:

            # Extract the vertex with minimum distance value
            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            # Traverse through all adjacent vertices of u
            # (the extracted vertex) and update their
            # distance values
            for pCrawl in self.graph[u]:

                v = pCrawl[0]

                # If shortest distance to v is not finalized
                # yet, and distance to v through u is less than
                # its previously calculated distance
                if minHeap.isInMinHeap(v) and pCrawl[1] < key[v]:
                    key[v] = pCrawl[1]
                    parent[v] = u

                    # update distance value in min heap also
                    minHeap.decreaseKey(v, key[v])


class DjikstrasShortestPath:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        """
        Utility printing function
        """
        print
        "Vertex \tDistance from Source"
        for node in range(self.V):
            print
            node, "\t", dist[node]


    def minDistance(self, dist, sptSet):
        """
        A utility function to find the vertex with minimum distance value, from the set of vertices not yet included
        in shortest path tree
        """
        # Initialize minimum distance for next node
        m = sys.maxint

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < m and not sptSet[v]:
                m = dist[v]
                min_index = v

        return min_index

    def dijkstra(self, src):
        """
        Function that implements Dijkstra's single source shortest path algorithm for a graph represented using
        adjacency matrix representation
        """
        dist = [sys.maxint] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from the set of vertices not yet processed u is always equal to src
            # in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the shortest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices of the picked vertex only if the current
            # distance is greater than new distance and the vertex in not in the shortest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                        dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        self.printSolution(dist)


class UnionFindWithRank:

    class Graph:
        def __init__(self, num_of_v):
            self.num_of_v = num_of_v
            self.edges = defaultdict(list)

        # graph is represented as an
        # array of edges
        def add_edge(self, u, v):
            self.edges[u].append(v)

    class Subset:
        def __init__(self, parent, rank):
            self.parent = parent
            self.rank = rank

    # A utility function to find set of an element
    # node(uses path compression technique)
    def find(self, subsets, node):
        if subsets[node].parent != node:
            subsets[node].parent = self.find(subsets, subsets[node].parent)
        return subsets[node].parent

    # A function that does union of two sets
    # of u and v(uses union by rank)
    def union(self, subsets, u, v):
        # Attach smaller rank tree under root
        # of high rank tree(Union by Rank)
        if subsets[u].rank > subsets[v].rank:
            subsets[v].parent = u
        elif subsets[v].rank > subsets[u].rank:
            subsets[u].parent = v

        # If ranks are same, then make one as
        # root and increment its rank by one
        else:
            subsets[v].parent = u
            subsets[u].rank += 1

    # The main function to check whether a given
    # graph contains cycle or not

    def isCycle(self, graph):

        # Allocate memory for creating sets
        subsets = []

        for u in range(graph.num_of_v):
            subsets.append(self.Subset(u, 0))

        # Iterate through all edges of graph,
        # find sets of both vertices of every
        # edge, if sets are same, then there
        # is cycle in graph.
        for u in graph.edges:
            u_rep = self.find(subsets, u)

            for v in graph.edges[u]:
                v_rep = self.find(subsets, v)

                if u_rep == v_rep:
                    return True
                else:
                    self.union(subsets, u_rep, v_rep)


class JobSequenceProblemDisjointSet:
    """
    Given a set of n jobs where each job i has a deadline di >=1 and profit pi>=0. Only one job can be scheduled at a
    time. Each job takes 1 unit of time to complete. We earn the profit if and only if the job is completed by its
    deadline. The task is to find the subset of jobs that maximizes profit.
    """

    class DisjointSet:
        def __init__(self, n):
            self.parent = [i for i in range(n + 1)]

        def find(self, s):
            # Make the parent of nodes in the path from
            # u --> parent[u] point to parent[u]
            if s == self.parent[s]:
                return s
            self.parent[s] = self.find(self.parent[s])
            return self.parent[s]

        # Make u as parent of v
        def merge(self, u, v):
            # Update the greatest available
            # free slot to u
            self.parent[v] = u

    def cmp(self, a):
        return a['profit']

    def findmaxdeadline(self, arr, n):
        """
        :param arr: Job array
        :param n: length of array
        :return: maximum deadline from the set of jobs
        """
        ans = - sys.maxsize - 1
        for i in range(n):
            ans = max(ans, arr[i]['deadline'])
        return ans

    def printjobscheduling(self, arr, n):
        """
        Find the maximum deadline among all jobs and
        create a disjoint set data structure with
        max_deadline disjoint sets initially
        """
        # Sort jobs in descending order on
        # basis of their profit
        arr = sorted(arr, key=self.cmp, reverse=True)

        max_deadline = self.findmaxdeadline(arr, n)
        ds = self.DisjointSet(max_deadline)

        for i in range(n):
            # find maximum available free slot for
            # this job (corresponding to its deadline)
            available_slot = ds.find(arr[i]['deadline'])
            if available_slot > 0:
                # This slot is taken by this job 'i'
                # so we need to update the greatest free slot.
                # Note: In merge, we make first parameter
                # as parent of second parameter.
                # So future queries for available_slot will
                # return maximum available slot in set of
                # "available_slot - 1"
                ds.merge(ds.find(available_slot - 1),
                         available_slot)
                print(arr[i]['id'], end=" ")


class AStar:
    """
    To approximate the shortest path in real-life situations, like- in maps, games where there can be many hindrances.
    We can consider a 2D Grid having several obstacles and we start from a source cell to reach towards a goal cell
    """