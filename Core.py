import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import networkx as nx
import numpy as np
from collections import OrderedDict
from threading import Lock
import scipy.sparse as sparse
from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.lin_ops.lin_utils as lu
from tqdm import tqdm


class CustomGraph(nx.DiGraph):
    """
    Custom Graph Initialization
    This Class is able to implement both simple path based and shortest path based initialization
    simple path : between every two source and target nodes we will consider all paths between them
    shortest path : between every two nodes we only take part those that are shortest . this method has a drawback that
    is we guessed that at the end not all routes that are being used is from these paths but the advantage is obviously
    memory!
    """

    def __init__(self, m, n, only_shortest_paths=True):
        super(CustomGraph, self).__init__()
        self.name = "my_grid_graph"
        self.m = m
        self.n = n
        self.only_shortest_path = only_shortest_paths
        self.__grid_construction(m, n)
        # Position Of Each Node
        xgrid, ygrid = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))
        position = [np.array((xgrid[key], ygrid[key])) for key in self.nodes.keys()]
        # node positions
        self.pos = OrderedDict(zip(self.nodes.keys(), position))
        # specify edge flows (protected)
        self._y = np.zeros((len(self.edges.keys()),))
        self.isEquilibrium = False
        # set additional attributes to graph
        self.__set_attributes()

        self.H = 0
        self.A = 0
        self.AtA = 0
        self.A_tild = 0
        self.C_tild = 0
        self.C_tild_squared = []
        self.nodes_dictionary = {}
        self.edges_dictionary = {}
        self.path_dictionary = {}
        self.total_number_of_active_paths = []
        self.number_of_paths = {}
        self.number_of_edges = {}
        self.pairs_distances = {}
        self.mapping = None
        self.__add_matrix()

        self.average_trip_cost = 0

        # The Most Important Thing
        self.__x = np.zeros((self.A.shape[1],))
        self.x_predecessor = self.__x.copy()

        # Demand Creation
        self.Demand = None
        self.__generate_demand()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, vector):
        # Customized Setter
        if self.isEquilibrium:
            raise ValueError("In User Equilibrium , we can't change flow on links !")
        else:
            self._y = vector

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, vector):
        self.__x = vector

    def get(self, name):
        return nx.get_edge_attributes(self, name)

    def __grid_construction(self, m, n):
        """ Private method to construct grid"""
        rows = range(m)
        columns = range(n)
        self.add_nodes_from((i, j) for i in rows for j in columns)
        self.add_edges_from(((i, j), (i - 1, j)) for i in rows for j in columns if i > 0)
        self.add_edges_from(((i, j), (i, j - 1)) for i in rows for j in columns if j > 0)
        self.add_edges_from(((i, j), (i + 1, j)) for i in rows for j in columns if i < m - 1)
        self.add_edges_from(((i, j), (i, j + 1)) for i in rows for j in columns if j < n - 1)

    def __set_attributes(self):
        nx.set_edge_attributes(self, dict(zip(self.edges.keys(), np.zeros((len(self.edges.keys()),)))), 'weight')
        nx.set_edge_attributes(self, dict(zip(self.edges.keys(), self.y)), 'flow')
        nx.set_edge_attributes(self, dict(zip(self.edges.keys(), (lambda x: 1 + x ** 2,) * len(self.y))), 'Delay Func')
        nx.set_edge_attributes(self, dict(zip(self.edges.keys(), range(len(self.edges.keys())))), 'index')
        nx.set_node_attributes(self, dict(zip(self.nodes.keys(), range(len(self.nodes.keys())))), 'index')

    def __add_matrix(self):
        """
        Adding Incidence Matrices to class
        Consider that this method will automatically generate H matrix with rows are flattened indexes of OD matrix
        """
        edges_dictionary = nx.get_edge_attributes(self, 'index')
        nodes_dictionary = nx.get_node_attributes(self, 'index')

        pairs = ((i, j) for i in self.nodes.keys() for j in self.nodes.keys())
        sorted_index = np.array([nodes_dictionary[first_node] * len(self.nodes) + nodes_dictionary[second_node]
                                 for first_node in self.nodes.keys()
                                 for second_node in self.nodes.keys()])
        self.mapping = np.argsort(sorted_index)
        # dictionary of generators for all paths
        paths_generator = {}
        paths_dict = {}
        # number of paths between pair of nodes
        number_of_paths = {}

        pair_path_indptr = [0]
        link_path_indptr = [0]
        data = []
        data_tild = []
        C_tild = []
        link_path_indices = []
        path_index = 0
        link_path_index = 0

        # this will make l_1 distance between nodes
        p = OrderedDict(nx.all_pairs_shortest_path_length(self))
        pairwise_dist = {(first_node, second_node): 1 + p[first_node][second_node] for first_node in p
                         for second_node in p[first_node]}

        for pair in tqdm(pairs):
            if pair[0] != pair[1]:
                if self.only_shortest_path:
                    paths_generator[pair] = nx.all_shortest_paths(self, pair[0], pair[1])
                else:
                    paths_generator[pair] = nx.all_simple_paths(self, pair[0], pair[1])

                for path in paths_generator[pair]:
                    data.append(pairwise_dist[pair])
                    paths_dict[tuple(path)] = path_index
                    link_path_indices.extend([edges_dictionary[key] for key in zip(path[:-1], path[1:])])
                    data_tild.extend([1 / float(pairwise_dist[pair]) for _ in range(len(path) - 1)])
                    link_path_index += len(path) - 1
                    link_path_indptr.append(link_path_index)
                    C_tild.append(1 / float(pairwise_dist[pair]))
                    path_index += 1

                number_of_paths[pair] = path_index - pair_path_indptr[-1]
                data_tild[pair_path_indptr[-1]:] /= np.sqrt(number_of_paths[pair])
                C_tild[-1] /= np.sqrt(number_of_paths[pair])
                pair_path_indptr.append(path_index)

            else:
                number_of_paths[pair] = path_index - pair_path_indptr[-1]
                # Add a zero row to H
                pair_path_indptr.append(path_index)

        pair_path_indices = range(path_index)

        self.H = sparse.csr_matrix((data, pair_path_indices, pair_path_indptr))[self.mapping, :]
        # the columns of this matrix have the same mapping of paths to index ,with H
        self.A = sparse.csc_matrix((np.ones((len(link_path_indices),)), link_path_indices, link_path_indptr))
        self.AtA = self.A.T.dot(self.A)
        self.A_tild = sparse.csc_matrix((data_tild, link_path_indices, link_path_indptr))
        # this is the vector containing of d_ij * sqrt(n_ij)
        self.C_tild = np.array(C_tild)
        self.C_tild_squared = self.C_tild ** 2
        self.nodes_dictionary = nodes_dictionary
        self.edges_dictionary = edges_dictionary
        self.total_number_of_active_paths = self.H.shape[1]
        self.number_of_paths = number_of_paths
        self.number_of_edges = self.A.shape[0]
        self.pairs_distances = pairwise_dist
        self.path_dictionary = paths_dict

    def __generate_demand(self):
        # Hmmm !-------------------------
        np.random.seed(1)
        # -------------------------------
        node_dict = self.nodes_dictionary
        attraction = 10 * np.random.rand(self.m * self.n, 1) + 0.3
        demand = 10 * np.random.rand(1, self.m * self.n) + 0.3
        self.Demand = np.matmul(attraction, demand)
        self._real_delta = np.sqrt((attraction ** 2).sum() * (demand ** 2).sum())

        for firstnode in self.nodes.keys():
            for secondnode in self.nodes.keys():
                if not firstnode == secondnode:
                    self.Demand[node_dict[firstnode], node_dict[secondnode]] /= self.pairs_distances[
                        (firstnode, secondnode)]

    def plot(self, ax=None):
        p = nx.single_source_shortest_path_length(self, (0, 0))

        nx.draw_networkx_edges(self, self.pos, nodelist=[(0, 0)], alpha=0.8, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(self, self.pos, nodelist=p.keys(), node_color=p.values(),
                               node_size=100, cmap=plt.cm.RdBu_r, ax=ax)
        nx.draw_networkx_labels(self, self.pos, dict(zip(self.nodes.keys(), range(len(self.nodes)))),
                                font_size=10)
        plt.axis('equal')
        plt.axis('off')
        # at the end you must put plt.show() yourself!


# This is Unusable for now . because i've changed the mapping so no need for complicated reshaping !
class reshape(AffAtom):
    """ This Class Will be Compatible with CVX and reshape data such that the diagonal is zero
    but also compatible with H indexing
    """

    def __init__(self, expr, nodes_dictionary, nodes):
        self.nodes = nodes
        self.nodes_dict = nodes_dictionary
        self.rows = len(nodes)
        self.cols = len(nodes)
        super(reshape, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Reshape the value.
        """
        output = np.zeros((self.rows, self.cols), values[0].dtype, "F")
        index = 0
        for first_node in self.nodes:
            for second_node in self.nodes:
                if not first_node == second_node:
                    output[self.nodes_dict[first_node], self.nodes_dict[second_node]] = values[0][index]
                    index += 1

        return output

    def validate_arguments(self):
        """Checks that the new shape has the same number of entries as the old.
        """
        old_len = self.args[0].size[0] * self.args[0].size[1]
        new_len = self.rows * self.cols - self.rows
        if not old_len == new_len:
            raise ValueError(
                "Invalid reshape dimensions (%i, %i)." % (self.rows, self.cols)
            )

    def size_from_args(self):
        """Returns the shape from the rows, cols arguments.
        """
        return (self.rows, self.cols)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.rows, self.cols]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Convolve two vectors.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.reshape(arg_objs[0], size), [])


class UESolver(object):

    def __init__(self, graph, Lambda=0.01, slowVersion=False):
        """
        this class will solve User Equilibrium Problem For a Custom Graph
        :param graph: a CustomGraph instance

        """
        assert isinstance(graph, CustomGraph)
        self.graph = graph
        # Step Size Parameters
        self.Lambda = Lambda
        # Trip Cost
        self.average_trip_cost = 0
        self.__slowVersion = slowVersion

        if self.__slowVersion:
            self.flow_on_routes = graph.x.copy()
        # Scope Instance
        self.scope = Scope(10, 0.01, self.update, 10000)

    def update(self, frame):

        if frame == 0:
            Lambda = 1.0
        else:
            Lambda = self.Lambda
        # Load Things !
        node_dict = self.graph.nodes_dictionary
        delayFuncs = self.graph.get('Delay Func')
        flows = OrderedDict(zip(self.graph.edges.keys(), np.zeros(len(delayFuncs))))
        path_dict = {}
        if self.__slowVersion: path_dict = self.graph.path_dictionary

        # Process
        paths = OrderedDict(nx.all_pairs_dijkstra_path(self.graph))
        self.flow_on_routes *= (1 - Lambda)

        for firstnode in paths:
            for secondnode in paths[firstnode]:
                path = paths[firstnode][secondnode]
                links_in_path = zip(path[:-1], path[1:])
                # Try to assign flow on routes
                try:
                    self.flow_on_routes[path_dict[tuple(path)]] += \
                        Lambda * self.graph.Demand[node_dict[firstnode], node_dict[secondnode]]
                except:
                    if self.__slowVersion and len(links_in_path) > 0:
                        print("Couldn't be able to assign flows on routes !")
                # Try to assign flow on links
                for links in links_in_path:
                    flows[links] += self.graph.Demand[node_dict[firstnode], node_dict[secondnode]]

        # PreProcess Before Assignment
        before_flows = self.graph.get('flow')
        flows = OrderedDict(
            [(key, (1 - Lambda) * before_flows[key] + Lambda * flows[key]) for key in self.graph.edges.keys()])
        new_wgts = np.array([delayFuncs[key](flows[key]) for key in self.graph.edges.keys()])
        new_weights = OrderedDict(zip(self.graph.edges.keys(), new_wgts))

        nx.set_edge_attributes(self.graph, new_weights, 'weight')
        nx.set_edge_attributes(self.graph, flows, 'flow')
        self.graph.y = np.array([flows[key] for key in self.graph.edges.keys()])
        self.average_trip_cost = (flows.values() * new_wgts).mean()

        return self.average_trip_cost

    def solve(self):
        self.scope.startFlag = True

    def reshape(self, values, distance_effect=False):
        if distance_effect:
            distances = np.array([self.graph.pairs_distances[(first_node, second_node)]
                                  for first_node in self.graph.nodes.keys()
                                  for second_node in self.graph.nodes.keys()], dtype=np.float)[self.graph.mapping]
            return np.reshape(values / distances, newshape=self.graph.Demand.shape)
        else:
            return np.reshape(values, newshape=self.graph.Demand.shape)

    def save_output(self):
        """
        this will generate a mat file that contains A,H,x,demand !
        also there will be a mapping that gives the correspondence between rows of H and pairs so that you could
        be able to reshape Hx . the value in the i'th index shows the row and column index of this flattened vector
        """
        from scipy import io
        # Create Mapping
        mapping = np.array([(self.graph.nodes_dictionary[first_node], self.graph.nodes_dictionary[second_node])
                            for first_node in self.graph.nodes.keys()
                            for second_node in self.graph.nodes.keys()
                            if not first_node == second_node])
        mdict = {'A': self.graph.A, 'H': self.graph.H, 'x_true': self.flow_on_routes, 'demand': self.graph.Demand,
                 'mapping': mapping}

        io.savemat('UE_checkpoint', mdict, oned_as='column', format='4')


class Scope(object):

    def __init__(self, maxt, dt, updateFcn, nIteration):
        """
        This Class is responsible to draw live animation
        :type updateFcn: callable function for update . this function should return a scalar value and
        should only take on frame argument
        :param ax: Axis object
        :param maxt: Maximum Time Span To show In PLot
        :param dt: Delta t
        :param nIteration : number of iterations
        """
        self.__fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.ax.relim()
        self.ax.autoscale_view(True, False, True)
        self.text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.dt = dt
        self.maxt = maxt
        self.x_data = [0]
        self.y_data = [0]
        # Set Axis
        self.line = plt.Line2D(self.x_data, self.y_data)
        self.ax.add_line(self.line)
        # Set Limits
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.maxt)
        # Update Fcn
        assert callable(updateFcn)
        self.__updatefcn = updateFcn
        # Threading
        self.mute = Lock()

        # animation Fucntion
        self.IterLimit = nIteration
        self.__anim = animation.FuncAnimation(self.__fig, self.update, frames=nIteration, blit=True, interval=100,
                                              repeat=False)

        # Start Plotting
        self.__startFlag = False

    @property
    def startFlag(self):
        return self.__startFlag

    @startFlag.setter
    def startFlag(self, boolvar):
        if isinstance(boolvar, bool) and boolvar:
            self.__startFlag = True
            plt.show()
        else:
            raise ValueError("start flag is a boolean variable")

    def RangeCorrection(self):
        # X Limit correction
        if self.x_data[-1] > self.maxt + self.x_data[0]:
            self.x_data = [self.x_data[-1]]
            self.y_data = [self.y_data[-1]]
            self.ax.set_xlim(self.x_data[0], self.x_data[0] + self.maxt)
            self.ax.set_ylim(self.y_data[-1] * 0.8, self.y_data[-1] * 1.2)
            self.ax.figure.canvas.draw()
        # Y Limit correction
        y_lims = self.ax.get_ylim()
        if self.y_data[-1] > y_lims[1]:
            self.ax.set_ylim(y_lims[0], self.y_data[-1] * 1.1)
        if self.y_data[-1] < y_lims[0]:
            self.ax.set_ylim(self.y_data[-1] * 0.9, y_lims[1])

    def update(self, frame):

        with self.mute:
            newValue = self.__updatefcn(frame)
            self.x_data.append(self.x_data[-1] + self.dt)
            self.y_data.append(newValue)
            self.RangeCorrection()
            self.line.set_data(self.x_data, self.y_data)
            self.text.set_text('objective = %f' % (self.y_data[-1]))
            plt.draw()

        return self.line, self.text

    def __del__(self):
        plt.close(self.__fig)
        del self
