import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gp.base.base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, sess, model, data, config):
        super(Trainer, self).__init__(sess, model, data, config, None)

    def train(self):

        initial_lstm_state = np.zeros((2, self.config.batch_size, self.config.lstm_size))

        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            cur_iterations = 0
            losses = []
            cur_epoch = self.cur_epoch_tensor.eval(self.sess)
            loop = tqdm(self.data.next_batch(), total=self.config.nit_epoch, desc="epoch-" + str(cur_epoch) + "-")

            for batch_x, batch_y, batch_actions, batch_rewards, new_sequence in loop:

                # Update the Global step
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

                if new_sequence:
                    feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: initial_lstm_state, self.model.is_training: True}
                    last_state = self.sess.run(self.model.final_lstm_state, feed_dict)

                else:
                    feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.actions: batch_actions,
                                 self.model.rewards: batch_rewards,
                                 self.model.initial_lstm_state: last_state, self.model.is_training: True}
                    out, _, loss, last_state = self.sess.run(
                        [self.model.output, self.model.train_step, self.model.loss,
                         self.model.final_lstm_state], feed_dict)
                    losses.append(loss)

                cur_iterations += 1
                # finish the epoch
                if cur_iterations >= self.config.nit_epoch:
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

            if cur_epoch % self.config.test_every == 0:
                self.test(cur_it)

        print("Training Finished")

    def test(self, cur_it):
        initial_lstm_state = np.zeros((2, self.data.xtest.shape[0], self.config.lstm_size))

        feed_dict = {self.model.x: self.data.xtest[:, :self.config.truncated_time_steps],
                     self.model.y: self.data.ytest[:, :self.config.truncated_time_steps],
                     self.model.rewards: self.data.rewardstest[:, :self.config.truncated_time_steps],
                     self.model.actions: self.data.actionstest[:, :self.config.truncated_time_steps],
                     self.model.initial_lstm_state: initial_lstm_state, self.model.is_training: False}
        last_state = self.sess.run(self.model.final_lstm_state, feed_dict)

        losses = []
        for i in range(1, int(self.config.episode_length / self.config.truncated_time_steps), 1):
            feed_dict = {self.model.x: self.data.xtest[:, i * self.config.truncated_time_steps:(
                                                                                                   i + 1) * self.config.truncated_time_steps],
                         self.model.y: self.data.ytest[:, i * self.config.truncated_time_steps:(
                                                                                                   i + 1) * self.config.truncated_time_steps],
                         self.model.actions: self.data.actionstest[:, i * self.config.truncated_time_steps:(
                                                                                                               i + 1) * self.config.truncated_time_steps],
                         self.model.rewards: self.data.rewardstest[:, i * self.config.truncated_time_steps:(
                                                                                                               i + 1) * self.config.truncated_time_steps],
                         self.model.initial_lstm_state: last_state, self.model.is_training: False}
            out, loss, last_state = self.sess.run(
                [self.model.output, self.model.loss, self.model.final_lstm_state], feed_dict)

            losses.append(loss)

        summaries_dict = {'test_MSE': np.mean(losses)}
        self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)
