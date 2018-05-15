from vinge.filters import id, logline, tag
from parser import *
from semex import SensorSemex, ConcatSemex, DisjunctSemex, StarSemex, TrivialSemex

def _base_absyn(graph, node):
    bt = node.base_type
    if bt == BaseType.ANYTHING:
        return TrivialSemex(graph.number_of_nodes())
    else:
        if bt == BaseType.LOGLINE:
            f = logline
        elif bt == BaseType.TAG:
            f = tag
        elif bt == BaseType.ID:
            f = id
        return SensorSemex(graph.number_of_nodes(),
                           f, graph)

def _concat_absyn(graph, transition, transition_op, node):
    regex1 = ast_to_semex(graph, transition, transition_op, node.regex1)
    regex2 = ast_to_semex(graph, transition, transition_op, node.regex2)
    return ConcatSemex(transition, transition_op, regex1, regex2)

def _disjunct_absyn(graph, transition, transition_op, node):
    regex1 = ast_to_semex(graph, transition, transition_op, node.regex1)
    regex2 = ast_to_semex(graph, transition, transition_op, node.regex2)
    return DisjunctSemex(regex1, regex2)

def _star_absyn(graph, transition, transition_op, node):
    rest = ast_to_semex(graph, transition, transition_op, node.regex)
    # TODO(trevor) length, what to do with that?
    return StarSemex(transition, transition_op,
                     graph.number_of_nodes(),
                     rest, length=3.0)

def ast_to_semex(graph, transition, transition_op, ast):
    """
    Converts the given regex abstract syntax tree into an actual regex to
    be used with the given graph.

    Args:
        graph (networkx.graph)
        transition (scipy.sparse.base.spmatrix) Adjacency matrix of graph
        transition_op (scipy.sparse.linalg.LinearOperator) LinOp of transition
        ast (regex_parser.RegExAbsSyn)
    Returns:
        (semex.Semex)
    """
    # TODO(trevor) should redo the regex stuff to not have to pass around these
    # three args (graph, transition, transition_op).
    if isinstance(ast, BaseAbsyn):
        return _base_absyn(graph, ast)
    elif isinstance(ast, ConcatAbsyn):
        return _concat_absyn(graph, transition, transition_op, ast)
    elif isinstance(ast, DisjunctAbsyn):
        return _disjunct_absyn(graph, transition, transition_op, ast)
    elif isinstance(ast, StarAbsyn):
        return _star_absyn(graph, transition, transition_op, ast)
    else:
        raise ValueError("Programmer error: This should never happen.")
