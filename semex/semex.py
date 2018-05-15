"""
A semex is:
sensor
semex | semex
semex *
semex semex

We wish to manipulate probability distributions and other weightings
over the nodes of a Markov chain. We also wish to represent
probability distributions and other weightings over sets of paths
through the Markov chain.

Weighted pathsets are naturally represented as linear operators over
distributions: consider a matrix A, where the entry A_{ij} gives the
total weight of paths in the path set which start at node #i and end
at node #j. Multiplying such matrices then correctly concatenates all
compatible paths from two path sets. 

ASIDE: linear algebra is a treacherous subject because it can be done
in two equivalent ways, with row vectors multiplied by matrices on
their right, or column vectors multiplied by matrices on their
left. For the sake of clarity, let us consider only row vectors. (Note
that this is only sort of clear; in particular, concatenation of
semexes is the reverse of the corresponding matrix multiplication.) In
the code below, we use both versions, because this lets us use
important functionality (linear operators) from scipy. We will put a
comment whenever we break this convention.

Semexes ("semiotic expressions") are an intuitive way to specify such
linear operators.

Semexes are implemented as follows. This can be thought of as a
weighted version of the Thompson NFA algorithm for implementing
regular expression search [1, but see 2 for an exposition intelligible
to the modern reader].

  - A sensor is a function f from nodes to R (the reals), representing
    a set of paths of length zero (i.e., paths containing a single
    node and no edges). As a linear operator, it sends d[i] to
    f(node_i) * d[i]. Sensors correspond to diagonal matrices.

    Sensors can be thought of as the emission probabilities of a
    time-dependent HMM.

  - Disjunction (semex | semex) is the union of two path sets. As a
    linear operator, this is implemented as the sum of the constituent
    linear operators. Note that this causes double counting if the
    choices overlap. We posit that such double counting is not a major
    concern, but users should be aware of this when crafting semiotic
    expressions.

  - The Kleene star (sensor semex length *) is the set of all paths of
    any length, weighted by the sensor at the nodes and the semex at
    the edges. For technical reasons, we require the user to specify a
    desired length. The semex will match paths of any length, but it
    will prefer paths of roughly the specified length.

  - Concatentation (semex sensor semex) represents the set of paths
    formable by joining any path from the first path set with any path
    from the second path set. The sensor specifies a weighting based
    on the node used to join the paths.

    This is implemented as the product of three linear operators: the
    linear operator of the first operand, the sensor operator, and the
    linear operator of the second operand.

Semexes can be compiled, either into a matrix (implemented as a scipy
sparse matrix when possible, and as a numpy matrix otherwise) or into
a scipy LinearOperator, which should be far more efficient. The
LinearOperators are the transposes of the matrices, since
LinearOperators are designed to be used like (x -> Ax) rather than (x
-> xA). The second fits more naturally with Markov chains and the
semantics of semiotic expressions.

References:

[1] Ken Thompson, "Regular expression search algorithm",
Communications of the ACM 11(6) (June 1968),
pp. 419-422. http://doi.acm.org/10.1145/363347.363387 (PDF)

[2] Russ Cox, "Regular Expression Matching Can Be Simple And Fast",
January 2007, http://swtch.com/~rsc/regexp/regexp1.html
"""

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator

# multiply linear operators
def mul(a,b):
    def matvec(v):
        return a.matvec(b.matvec(v))
    return LinearOperator(a.shape, matvec=matvec)

# add linear operators
def add(a,b):
    def matvec(v):
        return a.matvec(v) + b.matvec(v)
    return LinearOperator(a.shape, matvec=matvec)

# subtract linear operators
def sub(a,b):
    def matvec(v):
        return a.matvec(v) - b.matvec(v)
    return LinearOperator(a.shape, matvec=matvec)

# multiply linear operator by scalar
def scl(a,s):
    def matvec(v):
        return s * a.matvec(v)
    return LinearOperator(a.shape, matvec=matvec)

class Semex(object):
    def compile_into_matrix(self):
        return NotImplemented

    def compile_into_linop(self):
        return NotImplemented

    def linop_calculate_values(self, length):
        """
        This should export the linear operator compiled in the
        'compile_into_linop' method as an array of floats.

        Args:
            length (int) Length of the return vector

        Returns:
            numpy.array
        """
        vals = np.ones(length)
        return self.compile_into_linop().matvec(vals)

    def apply(self, dist):
        return NotImplemented

    def __str__(self):
        """
        Implementors of this class must export a valid string representation
        this semex ie:
          >>> from parse_regex import compile_regex
          >>> from ast_to_semex import ast_to_semex
          >>> semex1 = ast_to_semex(compile_regex(".*"))
          >>> semex2 = ast_to_semex(compile_regex(str(semex1)))
          >>> semex1 == semex2
        """
        return NotImplemented

class TrivialSemex(Semex):
    def __init__(self, nnodes):
        self.nnodes = nnodes
        self._cached_linop = None

    def compile_into_matrix(self):
        return sp.sparse.eye(self.nnodes, self.nnodes)

    def compile_into_linop(self):
        if self._cached_linop:
            return self._cached_linop
        self._cached_linop = LinearOperator((self.nnodes, self.nnodes), matvec=self.apply)
        return self._cached_linop

    def apply(self, dist):
        return dist.copy()

    def __cmp__(self, other):
        if not isinstance(other, TrivialSemex):
            return cmp(self, other)
        return cmp(self.nnodes, other.nnodes)

    def __str__(self):
        return '.'

class SensorSemex(Semex):
    def __init__(self, nnodes, thesensor, graph):
        ''' thesensor: maps integer ids to reals
        '''
        self.nnodes = nnodes 
        self.thesensor = thesensor
        self.graph = graph
        self._sensor_vector = self._create_sensor_vector()
        self._cached_linop = None

    def _create_sensor_vector(self):
        """
        Applies self.thesensor to self.graph to make a vector of the results.
        Returns: numpy.array
        """
        vector = np.zeros(self.graph.number_of_nodes())
        for i, node in enumerate(self.graph.nodes_iter()):
            vector[i] = self.thesensor(node)
        return vector

    def compile_into_matrix(self):
        sensor_values = np.zeros(self.nnodes)
        for i in xrange(self.nnodes):
            sensor_values[i] = self.thesensor(i)
        return sp.sparse.spdiags(sensor_values, [0], self.nnodes, self.nnodes)

    def compile_into_linop(self):
        if self._cached_linop:
            return self._cached_linop
        self._cached_linop = LinearOperator((self.nnodes,self.nnodes), matvec=self.apply)
        return self._cached_linop

    def apply(self, dist):
        # self._sensor_vector is a numpy array. '*' does element-wise
        # multiplication
        return dist * self._sensor_vector

    def __cmp__(self, other):
        if not isinstance(other, SensorSemex):
            return cmp(self, other)
        else:
            return cmp((self.nnodes, self.thesensor, self.graph),
                       (other.nnodes, other.thesensor, other.graph))

    def __str__(self):
        return "%s" % self.thesensor.__name__

class MatrixSensorSemex(Semex):
    def __init__(self, nnodes, thesensor, graph):
        ''' thefilter: vector
        This is a convenience. It provides the same functionality
        as SensorSemex, excpet the user provides a matrix instead of a function
        '''
        # TODO(trevor) clean this up. Probably merge MatrixFilter with Filter?
        self.nnodes = nnodes
        self.thesensor = thesensor
        self.graph = graph
        self._cached_linop = None

    def compile_into_matrix(self):
        sensor_values = np.zeros(self.nnodes)
        for i in xrange(self.nnodes):
            sensor_values[i] = self.thesensor(i)
        return sp.sparse.spdiags(sensor_values, [0], self.nnodes, self.nnodes)

    def compile_into_linop(self):
        if self._cached_linop:
            return self._cached_linop
        self._cached_linop = LinearOperator((self.nnodes,self.nnodes), matvec=self.apply)
        return self._cached_linop

    def apply(self, dist):
        # We want element-wise multiplication
        # self.thesensor is an array
        return self.thesensor * dist

    def __cmp__(self, other):
        if not isinstance(other, MatrixSensorSemex):
            return cmp(self, other)
        else:
            return cmp((self.nnodes, self.thesensor, self.graph),
                       (other.nnodes, other.thesensor, other.graph))

#TODO: do we want to specify weights on the alternatives?
class DisjunctSemex(Semex):
    def __init__(self, poss1, poss2):
        self.poss1 = poss1
        self.poss2 = poss2
        self.nnodes = poss1.nnodes
        self._cached_linop = None

    def compile_into_matrix(self):
        mat1 = self.poss1.compile_into_matrix()
        mat2 = self.poss2.compile_into_matrix()
        return mat1 + mat2

    def compile_into_linop(self):
        if self._cached_linop:
            return self._cached_linop
        op1 = self.poss1.compile_into_linop()
        op2 = self.poss2.compile_into_linop()
        self._cached_linop = add(op1,op2)
        return self._cached_linop

    def apply(self, dist):
        dist1 = self.poss1.apply(dist)
        dist2 = self.poss2.apply(dist)
        return dist1 + dist2

    def __cmp__(self, other):
        if not isinstance(other, DisjunctSemex):
            return cmp(self, other)
        else:
            return cmp((self.poss1, self.poss2),
                       (other.poss1, other.poss2))

    def __str__(self):
        return "(%s|%s)" % (self.poss1, self.poss2)

class StarSemex(Semex):
    def __init__(self, transition, transition_op, nnodes, inside, length):
        self.transition = transition
        self.transition_op = transition_op
        self.nnodes = nnodes
        self.inside = inside
        self.length = length
        self.pstop = 1.0 / self.length
        self.pgo = 1.0 - self.pstop
        self._cached_linop = None

    def compile_into_matrix(self):
        # TODO: some of the intermediate matrices could be sparse, but
        # they're not right now

        # Let F be the node filter
        fmat = self.inside.compile_into_matrix()
        fmat = fmat.todense()

        # Let T be the transition matrix
        tmat = self.transition.todense()
        tmat = np.array(tmat)
        print 'tmat', tmat.shape

        # The matrix we want is:
        # X =  (1-p) F + (1-p)p FTF + (1-p)p^2 FTFTF + (1-p)p^3 FTFTFTF + ...
        #   = (1-p) F (I + p TF + p^2 TFTF + p^3 TFTFTF + ...)
        #   = (1-p) F (I - p TF)^{-1}
        # (here p is self.pgo)

        # But the last step only works if all eigenvalues of pTF have
        # absolute value < 1, and preferably a decent bit away from 1,
        # like < 0.9. So, let's check that.
        ptfmat = self.pgo * np.dot(tmat, fmat)
        ptfmat_ = np.matrix(ptfmat)
        eigenvals, _ = np.linalg.eig(ptfmat_)
        assert (abs(eigenvals) <= 0.9).all()

        # Let Y = (I - p TF)
        ymat = np.eye(self.nnodes, self.nnodes) - ptfmat
        yinv = np.array(np.linalg.inv(ymat))

        # Let X = (1-p) F Y^{-1}
        xmat = self.pstop * np.dot(fmat, yinv)

        xmat = xmat.A

        return xmat

    def compile_into_linop(self):
        if self._cached_linop:
            return self._cached_linop

        # This is not commented very much, look at compile_into_matrix
        # for explanation. But note that linear operators are
        # transposed, so the details are a little different.

        fop = self.inside.compile_into_linop()
        top = self.transition_op
        pftop = scl(mul(fop,top), self.pgo)

        # TODO add eigenvalue check???

        # Y = (I - p FT)
        eye = aslinearoperator(sp.sparse.eye(self.nnodes, self.nnodes))
        yop = sub(eye, pftop)

        # guess Y^{-1} = (I + p FT) to speed up convergence below
        # TODO: maybe make a better guess? 
        precond = add(eye, pftop)

        def matvec(v):
            # ultimately want (1-p) Y^{-1} F v = Y^{-1} * (1-p)F v
            # so let's just get (1-p) F v:
            v = fop.matvec(v)
            v = self.pstop * v

            # calculate x = Y^{-1} v
            # by using LGMRES iteration to solve linear
            # equation Y x = v
            x,info = sp.sparse.linalg.lgmres(yop, v, M=precond)
            # check that cg converged etc.
            assert(info == 0)
            return x

        self._cached_linop = LinearOperator((self.nnodes,self.nnodes), matvec=matvec)
        return self._cached_linop

    def apply(self, dist):
        linop = self.compile_into_linop()
        return linop.matvec(dist)

    def __cmp__(self, other):
        if not isinstance(other, StarSemex):
            return cmp(self, other)
        else:
            return cmp((self.transition, self.transition_op, self.nnodes,
                        self.inside, self.length),
                       (other.transition, other.transition_op, other.nnodes,
                        other.inside, other.length))
    def __str__(self):
        return "(%s)*" % self.inside

class ConcatSemex(Semex):
    def __init__(self, transition, transition_op, part1, part2):
        self.transition = transition
        self.transition_op = transition_op
        self.part1 = part1
        self.part2 = part2
        self.nnodes = part1.nnodes
        # TODO(Trevor) this is jank. put assert part1.nnodes == part2.nnodes at least
        self._cached_linop = None

    def compile_into_matrix(self):
        mat1 = self.part1.compile_into_matrix() 
        mat2 = self.part2.compile_into_matrix()
        return mat1 * self.transition * mat2

    def compile_into_linop(self):
        if not self._cached_linop:
            op1 = self.part1.compile_into_linop()
            op2 = self.part2.compile_into_linop()

            # backwards because operators are transposed
            self._cached_linop = mul(op2, mul(self.transition_op, op1))
        return self._cached_linop

    def apply(self, dist):
        dist2 = self.part1.apply(dist)
        dist2 = dist2 * self.transition
        dist2 = self.part2.apply(dist2)
        return dist2

    def __cmp__(self, other):
        if not isinstance(other, ConcatSemex):
            return cmp(self, other)
        else:
            return cmp((self.transition, self.transition_op,
                        self.part1, self.part2),
                       (other.transition, other.transition_op,
                        other.part1, other.part2))
    def __str__(self):
        return "(%s %s)" % (self.part1, self.part2)
