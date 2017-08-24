import tensorflow as tf
from gp.layers.utils import mse, openai_entropy
from gp.utils.utils import find_trainable_variables, LearningRateDecay
from gp.a2c.models.base_model import BaseModel


class Model(BaseModel):
    def __init__(self, sess, policy, observation_space, action_space, num_envs, num_steps, num_stack,
                 entropy_coef=0.01, value_function_coeff=0.5, max_gradient_norm=0.5,
                 optimizer_params=None, total_timesteps=int(80e6),
                 lr_decay_method='linear'):
        self.num_actions = action_space.n
        self.batch_size = num_envs * num_steps
        self.num_steps = num_steps
        self.img_height, self.img_width, self.num_classes = observation_space.shape

        self.num_stack = num_stack
        self.X_input_train = None
        self.X_input_step = None
        self.actions = None
        self.advantage = None
        self.reward = None
        self.keep_prob = None
        self.is_training = None
        self.step_model = None
        self.train_model = None
        self.lr_decayer = None
        self.initial_state = None
        self.policy = policy
        self.sess = sess
        self.vf_coeff = value_function_coeff
        self.entropy_coeff = entropy_coef
        self.max_grad_norm = max_gradient_norm
        self.lr_decay_method = lr_decay_method
        self.total_timesteps = total_timesteps
        # RMSProp params = {'learning_rate': 7e-4, 'alpha': 0.99, 'epsilon': 1e-5}
        self.learning_rate = optimizer_params['learning_rate']
        self.alpha = optimizer_params['alpha']
        self.epsilon = optimizer_params['epsilon']

    def init_input(self):
        with tf.name_scope('input'):
            self.X_input_train = tf.placeholder(tf.uint8, (
                self.batch_size, self.img_height, self.img_width, self.num_classes * self.num_stack))
            self.X_input_step = tf.placeholder(tf.uint8, (
                self.batch_size // self.num_steps, self.img_height, self.img_width, self.num_classes * self.num_stack))

            self.actions = tf.placeholder(tf.int32, [self.batch_size])  # actions
            self.advantage = tf.placeholder(tf.float32, [self.batch_size])  # advantage function
            self.reward = tf.placeholder(tf.float32, [self.batch_size])  # reward

            self.learning_rate = tf.placeholder(tf.float32, [])  # learning rate
            self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob
            self.is_training = tf.placeholder(tf.bool)  # is_training

    def init_network(self):
        # The model structure
        self.step_model = self.policy(self.sess, self.X_input_step, self.num_actions, reuse=False,
                                      is_training=False)
        self.initial_state = self.step_model.initial_state

        self.train_model = self.policy(self.sess, self.X_input_train, self.num_actions, reuse=True,
                                       is_training=self.is_training)

        negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.policy_logits,
                                                                                  labels=self.actions)
        policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
        value_function_loss = tf.reduce_mean(mse(tf.squeeze(self.train_model.vf), self.reward))
        entropy = tf.reduce_mean(openai_entropy(self.train_model.pi))
        loss = policy_gradient_loss - entropy * self.entropy_coeff + value_function_loss * self.vf_coeff

        # Gradient Clipping
        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if self.max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        # Apply Gradients
        grads = list(zip(grads, params))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.alpha, epsilon=self.epsilon)
        optimize = optimizer.apply_gradients(grads)

        self.lr_decayer = LearningRateDecay(v=self.learning_rate, nvalues=self.total_timesteps,
                                            lr_decay_method=self.lr_decay_method)
