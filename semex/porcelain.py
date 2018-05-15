"""
Commands for interacting with semexes.
"""

from heapq import nlargest
from semex import ConcatSemex, MatrixSensorSemex

import numpy as np

def make_semex_starting_here(transition,
                             transition_op,
                             graph,
                             num_nodes,
                             semex,
                             node):
    """
    Make a new semex: node semex
    That is, start with the node passed in, followed by the semex.

    Args:
        transition (scipy.sparse.base.spmatrix) Adjacency matrix of graph
        transition_op (scipy.sparse.linalg.LinearOperator) LinOp of transition
        graph (networkx.DiGraph)
        num_nodes (int)
        semex (semex.semex.Semex)
        node (vertex.Vertex)

    Returns:
        semex.semex.Semex
    """
    state = np.zeros(num_nodes)
    state[node.idx()] = 1.0
    node_semex = MatrixSensorSemex(num_nodes, state, graph)
    new_semex = ConcatSemex(transition, transition_op, node_semex, semex)
    return new_semex

def most_likely_endpoints(semex, length, num_choose=4):
    """
    Calculates the most likely endpoints for the given semex.
    This assumes a uniform starting distribution.

    Args:
        semex (semex.semex.Semex)
        length (int)
        num_choose (int) the number of endpoints to return

    Returns:
        list of (int, float) - the index of the node in the graph's nodes array
            the float is the probability of the endpoint.
    """
    values = semex.linop_calculate_values(length)
    most_likely = nlargest(num_choose, enumerate(values),
                           key=lambda p: p[1])
    # TODO(trevor) num_choose should be able to be 'None' in which case
    # this returns the entire sorted list
    return most_likely
