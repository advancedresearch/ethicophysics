import tensorflow as tf

def build_computation(
        initial_beliefs,
        initial_aesthetics,
        initial_faiths,
        num_facts = 100,
        num_soul_axes = 4,
        num_leaders = 5,  # Churchill, FDR, Stalin vs. Hitler, Hirohito
        num_intellectuals = 10,
        num_preachers = 10,
        num_artists = 10,
        num_soldiers = 1000,
        num_cattle = 5000,
        alpha_truth=1e-3,
        alpha_religion=1e-3,
        alpha_art=1e-3):

    num_actors = (
        num_leaders +
        num_intellectuals +
        num_preachers +
        num_artists +
        num_soldiers +
        num_cattle)
    leader_start = 0
    leader_end = intellectual_start = num_leaders
    intellectual_end = preacher_start = (intellectual_start +
                                         num_intellectuals)
    preacher_end = artist_start = (preacher_start + num_preachers)
    artist_end = soldier_start = (artist_start + num_artists)
    soldier_end = cattle_start = (soldier_start + num_soldiers)
    cattle_end = (cattle_start + num_cattle)
    
    alive = tf.placeholder(
        tf.float32,
        name='alive',
        shape=num_actors)
    old_affiliations = tf.placeholder(
        tf.float32,
        name='old_affiliations',
        shape=(num_actors, num_soul_axes),
    )
    old_unskilled_skilled = old_affiliations[:, 0]

    beliefs = tf.get_variable(
        'beliefs',
        # shape=(num_actors, num_facts),
        initializer=initial_beliefs,
    )
    # for this naive model, we assume that all evidence is available
    # to all actors
    evidence = tf.placeholder(
        tf.float32,
        name='evidence',
        shape=num_facts)
    implications = tf.placeholder(
        tf.float32,
        name='implications',
        shape=(num_facts, num_soul_axes)
    )
    twinges = tf.reduce_mean(
        tf.reshape(evidence, (-1, 1)) * implications,
        axis=0,
        keepdims=True,
    )
    
    faith = tf.get_variable(
        'faith',
        # shape=(num_actors, num_preachers),
        initializer=initial_faiths,
    )
    # sermon effects are assumed to be saturating, and otherwise
    # a simple product of the skill and affiliation of the preacher
    sermon = tf.tanh(
        tf.reshape(old_unskilled_skilled[preacher_start:preacher_end],
                   (-1, 1)) *
        old_affiliations[preacher_start:preacher_end]
    )
    sermon_effects = tf.matmul(
        faith,
        sermon)
    
    aesthetics = tf.get_variable(
        'aesthetics',
        # shape=(num_actors, num_artists),
        initializer=initial_aesthetics,
    )
    # beauty is assumed to be saturating, and otherwise a rather
    # straightforward product of the taste of the viewer and the skill
    # of the artist
    beauty = tf.tanh(
        aesthetics *
        tf.reshape(old_unskilled_skilled[artist_start:artist_end],
                   (1, -1))
    )
    # movedness is assumed to be saturating, and otherwise a rather
    # straightforward product of the beauty of the art and the
    # affiliation of the artist
    movedness = tf.tanh(
        tf.matmul(beauty, 
                  old_affiliations[artist_start:artist_end, :]))
    
    new_affiliations = (
        # affiliations change slowly if at all
        old_affiliations +
        # changes in affiliation should accord with twinges
        alpha_truth * twinges +
        # changes in affiliation should accord with religious beliefs
        alpha_religion * sermon_effects +
        # changes in affiliation should accord with movedness
        alpha_art * movedness
    )        

    # people don't like changing their minds
    loss = tf.nn.l2_loss(new_affiliations - old_affiliations)

    optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)
    
    return dict(
        input=dict(
            old_affiliations=old_affiliations,
            alive=alive,
            evidence=evidence,
            implications=implications,
        ),
        output=dict(
            # life and affiliations
            alive=alive,
            old_affiliations=old_affiliations,
            new_affiliations=new_affiliations,

            # truth
            beliefs=beliefs,
            evidence=evidence,
            implications=implications,
            twinges=twinges,

            # beauty
            aesthetics=aesthetics,
            beauty=beauty,
            movedness=movedness,

            # goodness
            faith=faith,
            sermon=sermon,
            sermon_effects=sermon_effects,
            
            # learning
            loss=loss,
            optimizer=optimizer,
        )
    )
            
