import tensorflow as tf

def sign(x):
    return tf.tanh(1e6 * x)

class GreaterFoolModel(object):
    def __init__(self, n=100, k=5):
        self.n = n
        self.k = k
        self.build_computation()

    def build_computation(self):
        self.agent_fragments = tf.get_variable('fragments', shape=[self.k, 1])
        self.valuation_weights = []
        self.valuation_biases = []
        for i in range(self.n):
            wt = tf.get_variable('valuation_weights_%s' % i, shape=[1, self.k])
            b = tf.get_variable('valuation_biases_%s' % i, shape=[1, 1])
            self.valuation_weights.append(wt)
            self.valuation_biases.append(b)
        self.valuation_weights_cat = tf.concat(self.valuation_weights, axis=0)
        self.valuation_biases_cat = tf.concat(self.valuation_biases, axis=0)

        self.old_valuation = tf.placeholder(tf.float32)

        # add old_valuation to it so that people start off roughly at the old valuation
        self.valuation = (tf.matmul(self.valuation_weights_cat, self.agent_fragments) + 
                          self.valuation_biases_cat + self.old_valuation)

        print(self.valuation)


        self.new_valuation = tf.reduce_mean(self.valuation)

        # assume that people buy one unit if they think it will go up and short
        # one unit if they think it will go down
        self.expected_rewards = ((self.new_valuation - self.old_valuation) * 
                                 sign(self.valuation - self.old_valuation))

        self.fragment_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(
            self.expected_rewards, var_list=self.agent_fragments)
        self.investor_optimizers = []
        for i in range(self.n):
            opt = tf.train.GradientDescentOptimizer(0.1).minimize(
                -self.expected_rewards[i], var_list=[self.valuation_weights[i], 
                                                     self.valuation_biases[i]])
            self.investor_optimizers.append(opt)

            print(i, 'optimizers')

    def simulate(self):
        sess = tf.Session()
        for trial in range(100):
            sess.run(tf.initialize_all_variables())

            valuation = 100.0
            for t in range(10):
                for i in range(100):
                    _, _, valuation = sess.run([self.fragment_optimizer, 
                                                self.investor_optimizers,
                                                self.new_valuation],
                                               feed_dict={self.old_valuation: valuation})
                    print(trial, t, valuation.mean())

if __name__ == '__main__':
    model = GreaterFoolModel()
    model.simulate()
