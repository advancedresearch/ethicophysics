import tensorflow as tf

def sign(x):
    return tf.tanh(1e2 * x)

class VotingModel(object):
    def __init__(self, n=100, k=5):
        self.n = n
        self.k = k
        self.build_computation()

    def build_computation(self):
        self.agent_fragments = tf.get_variable('fragments', shape=[self.k, 1])
        self.eagerness_weights = []
        self.eagerness_biases = []
        for i in range(self.n):
            wt = tf.get_variable('eagerness_weights_%s' % i, shape=[1, self.k])
            b = tf.get_variable('eagerness_biases_%s' % i, shape=[1, 1])
            self.eagerness_weights.append(wt)
            self.eagerness_biases.append(b)
        self.eagerness_weights_cat = tf.concat(self.eagerness_weights, axis=0)
        self.eagerness_biases_cat = tf.concat(self.eagerness_biases, axis=0)
        self.opinion_weights = tf.get_variable('opinion_weights', shape=[self.n, self.k])
        self.opinion_biases = tf.get_variable('opinion_biases', shape=[self.n, 1])

        print(self.eagerness_weights_cat, self.eagerness_biases_cat)

        self.eagerness = tf.sigmoid(
            tf.matmul(self.eagerness_weights_cat, self.agent_fragments) + self.eagerness_biases_cat)
        # self.opinion = tf.tanh(
        #     tf.matmul(self.opinion_weights, self.agent_fragments) + self.opinion_biases)
        self.opinion = tf.tanh(1e6 * self.opinion_biases)

        print(self.eagerness, self.opinion)

        self.expected_vote = tf.reduce_mean(self.eagerness * sign(self.opinion))
        self.outcome = sign(self.expected_vote)

        self.expected_rewards = -self.eagerness + 10 * self.outcome * self.opinion

        self.fragment_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(
            self.expected_rewards, var_list=self.agent_fragments)
        self.voter_optimizers = []
        for i in range(self.n):
            opt = tf.train.GradientDescentOptimizer(0.1).minimize(
                -self.expected_rewards[i], var_list=[self.eagerness_weights[i], 
                                                     self.eagerness_biases[i]])
            self.voter_optimizers.append(opt)

            print(i, 'optimizers')

    def simulate(self):
        sess = tf.Session()
        tot = 0.0
        for trial in range(100):
            sess.run(tf.initialize_all_variables())

            for i in range(1000):
                _, _, eagerness, x = sess.run([self.fragment_optimizer, 
                                               self.voter_optimizers,
                                               self.eagerness,
                                               self.agent_fragments])
                print(trial, i, x, eagerness.mean())
            tot += eagerness.mean()
            print(trial, eagerness, tot / (trial + 1))


if __name__ == '__main__':
    model = VotingModel()
    model.simulate()
