import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import gym

if __name__== "__main__":
    env = gym.make('CartPole-v1')
    env.seed(55)
    obs = env.reset()

    # 1. Specify the network architecture
    n_inputs = 4  # == env.observation_space.shape[0]
    n_hidden = 4  # it's a simple task, we don't need more than this
    n_outputs = 1 # only outputs the probability of accelerating left
    initializer = tf.variance_scaling_initializer()

    # 2. Build the neural network
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                            kernel_initializer=initializer)
    outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid,
                            kernel_initializer=initializer)

    # 3. Select a random action based on the estimated probabilities
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    init = tf.global_variables_initializer()

    n_max_steps = 1000

    with tf.Session() as sess:
        init.run()
        obs = env.reset()
        for step in range(n_max_steps):
            env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break

    env.close()