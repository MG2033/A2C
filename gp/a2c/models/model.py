import tensorflow as tf
from gp.layers.utils import mse, openai_entropy
from gp.utils.utils import find_trainable_variables
from gp.configs.a2c_config import A2CConfig


class Model:
    def __init__(self, sess, observation_space_params, num_actions,
                 entropy_coef=0.01, value_function_coeff=0.5, max_gradient_norm=0.5,
                 optimizer_params=None):
        self.num_actions = num_actions
        self.train_batch_size = A2CConfig.num_envs * A2CConfig.unroll_time_steps
        self.num_steps = A2CConfig.unroll_time_steps
        self.img_height, self.img_width, self.num_classes = observation_space_params

        self.num_stack = A2CConfig.num_stack
        self.actions = None
        self.advantage = None
        self.reward = None
        self.keep_prob = None
        self.is_training = None
        self.step_policy = None
        self.train_policy = None
        self.learning_rate_decayed = None
        self.initial_state = None
        self.X_input_step_shape = None
        self.X_input_train_shape = None
        self.policy_gradient_loss = None
        self.value_function_loss = None
        self.optimize = None
        self.entropy = None
        self.loss = None
        self.learning_rate = None
        self.policy = A2CConfig.policy_class
        self.sess = sess
        self.vf_coeff = value_function_coeff
        self.entropy_coeff = entropy_coef
        self.max_grad_norm = max_gradient_norm
        # RMSProp params = {'learning_rate': 7e-4, 'alpha': 0.99, 'epsilon': 1e-5}
        self.initial_learning_rate = optimizer_params['learning_rate']
        self.alpha = optimizer_params['alpha']
        self.epsilon = optimizer_params['epsilon']

    def init_input(self):
        with tf.name_scope('input'):
            self.X_input_train_shape = (
                None, self.img_height, self.img_width, self.num_classes * self.num_stack)
            self.X_input_step_shape = (
                None, self.img_height, self.img_width,
                self.num_classes * self.num_stack)

            self.actions = tf.placeholder(tf.int32, [None])  # actions
            self.advantage = tf.placeholder(tf.float32, [None])  # advantage function
            self.reward = tf.placeholder(tf.float32, [None])  # reward
            self.learning_rate = tf.placeholder(tf.float32, [])  # learning rate
            self.is_training = tf.placeholder(tf.bool)  # is_training

    def init_network(self):
        # The model structure
        self.step_policy = self.policy(self.sess, self.X_input_step_shape, self.num_actions, reuse=False,
                                       is_training=False)

        self.train_policy = self.policy(self.sess, self.X_input_train_shape, self.num_actions, reuse=True,
                                        is_training=self.is_training)

        with tf.variable_scope('train_output'):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_policy.policy_logits,
                labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
            self.value_function_loss = tf.reduce_mean(mse(tf.squeeze(self.train_policy.value_function), self.reward))
            self.entropy = tf.reduce_mean(openai_entropy(self.train_policy.policy_logits))
            self.loss = self.policy_gradient_loss - self.entropy * self.entropy_coeff + self.value_function_loss * self.vf_coeff

            # Gradient Clipping
            params = find_trainable_variables("policy")
            grads = tf.gradients(self.loss, params)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            # Apply Gradients
            grads = list(zip(grads, params))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.alpha,
                                                  epsilon=self.epsilon)
            self.optimize = optimizer.apply_gradients(grads)

    def build(self):
        self.init_input()
        self.init_network()
