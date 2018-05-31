"""This is a simple test script for now, we are not simulating the
economy or wartime at all.
"""


import numpy as np
import tensorflow as tf

import learning

def main(verbose=False, arthur_boost=0, num_trials=10, message=''):
    LARGENUM = 100
    
    num_facts = 1  # Arthur pulled the sword out of the stone (or did he?)
    num_soul_axes = 3  # unskilled-skilled, evil-good, peasant-noble
    num_leaders = 2  # Arthur, Mordred
    num_intellectuals = 2  # Merlin, Morgana La Fey
    num_preachers = 2
    num_artists = 2
    num_soldiers = 20
    num_cattle = 5000
    num_actors = (
        num_leaders +
        num_intellectuals +
        num_preachers +
        num_artists +
        num_soldiers +
        num_cattle)

    for trial in range(num_trials):

        initial_beliefs = np.random.random(num_actors).astype(np.float32)
        initial_faiths = np.random.random(
            (num_actors, num_preachers)).astype(np.float32)
        initial_aesthetics = np.random.random(
            (num_actors, num_artists)).astype(np.float32)

        initial_affiliations = -1 + 2 * np.random.random(
            (num_actors, num_soul_axes)).astype(np.float32)
        # give Arthur a boost to see how initial conditions affect
        # things
        initial_affiliations[:, 1] += arthur_boost
        # Arthur
        initial_affiliations[0, :] = (LARGENUM, LARGENUM, -LARGENUM)
        # Mordred
        initial_affiliations[1, :] = (LARGENUM, -LARGENUM, -LARGENUM)

        implications = np.array([[
            1,  # Arthur believes in excellence and cultivates it
            1,  # Arthur and Mordred both want to be king 
            -1,  # Arthur was initially a peasant -> fuck the nobles
        ]])

        with tf.Graph().as_default():
            computation = learning.build_computation(
                initial_beliefs=initial_beliefs,
                initial_faiths=initial_faiths,
                initial_aesthetics=initial_aesthetics,
                num_facts=num_facts,
                num_soul_axes=num_soul_axes,
                num_leaders=num_leaders,
                num_intellectuals=num_intellectuals,
                num_preachers=num_preachers,
                num_artists=num_artists,
                num_soldiers=num_soldiers,
                num_cattle=num_cattle)

            alive = np.ones(num_actors, dtype=np.float32)
            evidence = np.ones(num_facts, dtype=np.float32)
            affiliations = initial_affiliations

            session = tf.Session()
            session.run(tf.global_variables_initializer())
            for t in range(1000):
                inputs = computation['input']
                results = session.run(computation['output'],
                            feed_dict={
                                inputs['old_affiliations']: affiliations,
                                inputs['alive']: alive,
                                inputs['evidence']: evidence,
                                inputs['implications']: implications,
                            }
                )
                affiliations = results['new_affiliations']
                if verbose:
                    for k, v in results.items():
                        if v is None or not v.shape:
                            continue
                        if v.shape[0] == num_actors:
                            print(k, v.mean(axis=0))
                        else:
                            print(k, v)
                    print()

        print(message, affiliations[:, 1].mean())
                
if __name__ == '__main__':
    main(arthur_boost=0, message='Balanced')
    main(arthur_boost=0.5, message='+Arthur')
    main(arthur_boost=-0.5, message='+Mordred')
    main(arthur_boost=1.0, message='++Arthur')
    main(arthur_boost=-1.0, message='++Mordred')
