import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gp.base.base_train

class Trainer():
    def __init__(self, sess, model, data, config):
        super(Trainer, self).__init__(sess, model, data, config)


    def train(self):

        initial_lstm_state = np.zeros((2, self._config.batch_size, 512))

        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self._config.n_epochs + 1, 1):

            cur_iterations = 0
            losses = []
            cur_epoch = self.cur_epoch_tensor.eval(self.sess)
            loop = tqdm(self.data.next_batch(), total=self._config.nit_epoch, desc="epoch-" + str(cur_epoch) + "-")

            for batch_x, batch_y, batch_actions, batch_rewards, new_sequence in loop:

                # Update the Global step
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

                if new_sequence:
                    feed_dict = {self.model.X: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: initial_lstm_state, self.model.is_training: True}
                    last_state = self.sess.run(self.model.final_lstm_state, feed_dict)

                else:
                    feed_dict = {self.model.X: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: last_state, self.model.is_training: True}
                    out, _, loss, mean_relative_error, last_state = self.sess.run(
                        [self.model.output, self.model.train_step, self.model.loss, self.model.mean_relative_error,
                         self.model.final_lstm_state], feed_dict)
                    losses.append(loss)
                    # print(mean_relative_error / (2 * self._config.nit_epoch / 3))

                cur_iterations += 1
                # finish the epoch
                if cur_iterations >= self._config.nit_epoch:
                    break

            cur_it = self.global_step_tensor.eval(self.sess)
            loss = np.mean(losses)

            summaries_dict = {'loss': loss}
            self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)

            loop.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))

            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            self.save()

            if cur_epoch % self._config.test_every == 0:
                self.test(cur_it)

        print("Training Finished")

    def test(self, cur_it):
        initial_lstm_state = np.zeros((2, self.data.xtest.shape[0], 512))

        feed_dict = {self.model.X: self.data.xtest[:, :self._config.truncated_time_steps],
                     self.model.y: self.data.ytest[:, :self._config.truncated_time_steps],
                     self.model.rewards: self.data.rewardstest[:, :self._config.truncated_time_steps],
                     self.model.actions: self.data.actionstest[:, :self._config.truncated_time_steps],
                     self.model.initial_lstm_state: initial_lstm_state, self.model.is_training: False}
        last_state = self.sess.run(self.model.final_lstm_state, feed_dict)

        losses = []
        for i in range(1, int(self._config.all_seq_length / self._config.truncated_time_steps), 1):
            feed_dict = {self.model.X: self.data.xtest[:, i * self._config.truncated_time_steps:(
                                                                                                    i + 1) * self._config.truncated_time_steps],
                         self.model.y: self.data.ytest[:, i * self._config.truncated_time_steps:(
                                                                                                    i + 1) * self._config.truncated_time_steps],
                         self.model.actions: self.data.actionstest[:, i * self._config.truncated_time_steps:(
                                                                                                                i + 1) * self._config.truncated_time_steps],
                         self.model.rewards: self.data.rewardstest[:, i * self._config.truncated_time_steps:(
                                                                                                                i + 1) * self._config.truncated_time_steps],
                         self.model.initial_lstm_state: last_state, self.model.is_training: False}
            out, loss, last_state = self.sess.run(
                [self.model.output, self.model.loss, self.model.final_lstm_state], feed_dict)

            losses.append(loss)

        summaries_dict = {'test_MSE': np.mean(losses)}
        self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)
