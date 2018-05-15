import networkx as nx
import scipy as sp

import vinge.filters
from vinge.semex.ast_to_semex import ast_to_semex
from vinge.semex.parser import *
from vinge.semex.semex import *
from vinge.vertex import LogLineVertex, UniqueIDVertex

# Don't care about this graph structure at all.
# Just need something to plumb the tests together.
graph = nx.DiGraph()
v1 = LogLineVertex("hello", "hello", 1, "threadid", 6)
v2 = UniqueIDVertex("ok!")
graph.add_edge(v1, v2)
transition = nx.to_scipy_sparse_matrix(graph)
transition_op = sp.sparse.linalg.aslinearoperator(transition)

def do_ast_to_semex(ast):
    return ast_to_semex(graph, transition, transition_op, ast)

class TestRegexAstToSemex:

    def test_any(self):
        ast = BaseAbsyn(BaseType.ANYTHING)
        semex = do_ast_to_semex(ast)
        assert semex == TrivialSemex(graph.number_of_nodes())

    def test_logline(self):
        ast = BaseAbsyn(BaseType.LOGLINE)
        semex = do_ast_to_semex(ast)
        assert semex == SensorSemex(graph.number_of_nodes(),
                                   vinge.filters.logline,
                                   graph)

    def test_tag(self):
        ast = BaseAbsyn(BaseType.TAG)
        semex = do_ast_to_semex(ast)
        assert semex == SensorSemex(graph.number_of_nodes(),
                                    vinge.filters.tag,
                                    graph)

    def test_id(self):
        ast = BaseAbsyn(BaseType.ID)
        semex = do_ast_to_semex(ast)
        assert semex == SensorSemex(graph.number_of_nodes(),
                                    vinge.filters.id,
                                    graph)

    def test_concat(self):
        ast = ConcatAbsyn(BaseAbsyn(BaseType.LOGLINE), BaseAbsyn(BaseType.ID))
        semex = do_ast_to_semex(ast)
        answer = ConcatSemex(transition, transition_op,
                             SensorSemex(graph.number_of_nodes(),
                                         vinge.filters.logline,
                                         graph),
                             SensorSemex(graph.number_of_nodes(),
                                         vinge.filters.id,
                                         graph)
                             )
        assert semex == answer

    def test_disjunct(self):
        ast = DisjunctAbsyn(BaseAbsyn(BaseType.ID), BaseAbsyn(BaseType.TAG))
        semex = do_ast_to_semex(ast)
        answer = DisjunctSemex(SensorSemex(graph.number_of_nodes(),
                                           vinge.filters.id,
                                           graph),
                               SensorSemex(graph.number_of_nodes(),
                                           vinge.filters.tag,
                                           graph)
                               )
        assert semex == answer

    def test_star(self):
        ast = StarAbsyn(BaseAbsyn(BaseType.ANYTHING))
        semex = do_ast_to_semex(ast)
        answer = StarSemex(transition, transition_op,
                          graph.number_of_nodes(),
                          TrivialSemex(graph.number_of_nodes()),
                          length=3.0)
        assert semex == answer
