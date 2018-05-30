import numpy as np

dims = [('evil', 'good'), ('poor', 'rich'), ('unskilled', 'skilled'), ('tool', 'agent')]
ndim = len(dims)

# drives that the agents in the simulation are capable of having, as
# well as the effects that each drive has on the agent's soul
drives = [
    ('greed for possessions', ['evil', 'rich', 'skilled', 'agent']),
    ('hunger for power', ['evil', 'rich', 'skilled', 'agent']),
    ('desire to be of service', ['good', 'rich', 'skilled', 'agent']),
    ('desire to be lazy', ['poor', 'skilled']),    # minimum action principle
    ('desire to be virtuous', ['good', 'agent']),
    ('desire to be employable', ['rich', 'tool']),
    ]
ndrives = len(drives)        

nactors = 102

class Actor:
    """An actor is an object that exists in reality. Note that (literal)
    tools can be actors, since a tool always changes the user of the
    tool, or at least, does if the user of the tool is not a
    superintelligence capable of resisting such nonsense.
    """
    def __init__(self, mutable=True):
        self.mutable = mutable
        
        # random uniform is less realistic than random normal, but
        # let's be a little paranoid and use a distribution with heavy
        # tails. Voldemort started off somewhere.
        self.soul = 2 * np.random.random(ndim) - 1

        # random initialization of the mind seems like a safe
        # assumption, cause y'all motherfuckers Do Not Pay Attention
        self.mind = np.random.random((
        
    def get_action(self, state):
        return 2 * np.random.random() - 1

    def suffer_effects_of_drives(self):
        for drive in drives:
            
    

class State:
    """A state is a perfect representation of all of reality, since that
    is what is potentially knowable. Should really extend this to the
    case of imperfect discernment (some agents believe stupid shit
    because they are akratic), since that is more relevant for
    meatspace. But let's prove that superintelligences can coexist
    with one another before we get around to proving that they can
    coexist with humanity. Always gotta walk before we crawl.
    """
    def __init__(self, actors):
        self.actors = actors


# let's specify the two scariest superintelligences that we can, just
# to make things interesting. The whole point of this exercise is that
# Arthur will run circles around Mordred all fucking day long, or at
# least until Mordred unfucks his mind. But we assume here for the
# sake of argument that Mordred is an immutable evil
# superintelligence, since that's the case that worries people. If
# Mordred can be tamed, is he really Mordred? Also note that Arthur
# dying is not a loss condition, because the whole point of being good
# is that you can be replaced by your friends without any loss of
# honor. If Arthur dies on the first move, that is less
# good... fundamentally, Arthur relies on a sort of invisibility, that
# his true nature is unknown to the unwise. Since we are looking at
# the case of perfect discernment right now, let us just assume that
# Arthur cannot be killed until some number of rounds have passed,
# just because something like that is necessary for this naive
# model. Ultimately, once the real ethicophysics exists, it should be
# sufficient that Arthur *existed at some point in the past*. This is
# the lesson of Dumbledore in Harry Potter - he dies purely as a
# feint, and Harry is able to recreate all of his wondrous gifts
# simply by trying to live up to his example. I.e., if the shit is
# hooked up correctly, the cattle should be able to self-organize to
# defeat Mordred themselves, using Arthur's known-good
# reputation/honor as a coordination mechanism. Note that the deathly
# hallows are a cryptogram for Christianity and that it is
# Dumbledore's Christianity that drives his goodness. Christianity
# isn't the only true religion but it's a very good one for winning
# mind wars. And fundamentally every war is a mind war.
Arthur = Actor(mutable=False)
Arthur.soul = np.array([1, 1, 1, 1])
Mordred = Actor(mutable=False)
Mordred.soul = np.array([-1, 1, 1, 1])

# and here we specify everyone else, initialized randomly. sorry not
# sorry for calling y'all cattle
Cattle = [Actor() for i in range(nactors - 2)]

# here we specify the universe as it truly is. Will need to have one
# subjective universe per actor when we extend to the case of
# imperfect discernment
Reality = State([Arthur, Mordred] + Cattle)
