from collections import namedtuple

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from playground.policies.base import BaseModelMixin, Policy, Config
from playground.policies.memory import ReplayMemory
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class PPOPolicy(Policy, BaseModelMixin):

    def __init__(self, env, name, training=True, gamma=0.99, lam=0.95,
                 actor_layers=[64, 32], critic_layers=[128, 64], clip_norm=None, **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete), \
            "Current PPOPolicy implementation only works for discrete action space."

        self.lam = lam  # lambda for GAE.
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.clip_norm = clip_norm

    def act(self, state, **kwargs):
        probas = self.sess.run(self.actor_proba, {self.s: [state]})[0]
        action = np.random.choice(range(self.act_size), size=1, p=probas)[0]
        return action

    def _build_networks(self):
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='state')
        self.a = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.s_next = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='next_state')
        self.r = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done = tf.placeholder(tf.float32, shape=(None,), name='done_flag')

        self.old_logp_a = tf.placeholder(tf.float32, shape=(None,), name='old_logp_actor')
        self.R = tf.placeholder(tf.float32, shape=(None,), name='return')
        self.adv = tf.placeholder(tf.float32, shape=(None,), name='adv')

        # this is a policy with value
        with tf.variable_scope('ppo_model'):

            #latent state from which policy distribution parameters should be inferred
            self.latent = dense_nn(self.s, self.actor_layers[:-1], name='prob_dist')
            self.actor = dense_nn(self.latent, self.actor_layers[-1:] + [self.act_size],
                                name='actor')
            self.actor_proba = tf.nn.softmax(self.actor)
            a_ohe = tf.one_hot(self.a, self.act_size, 1.0, 0.0, name='action_ohe')
            self.logp_a = tf.reduce_sum(tf.log(self.actor_proba) * a_ohe,
                                        reduction_indices=-1, name='new_logp_actor')

            self.v = tf.squeeze(dense_nn(self.latent, self.actor_layers[-1:] + [1], name='value'))

            self.params = self.scope_vars('ppo_model')

    def _build_train_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.clip_range = tf.placeholder(tf.float32, shape=None, name='ratio_clip_range')

        with tf.variable_scope('policy_value_train'):
            #policy loss
            ratio = tf.exp(self.logp_a - self.old_logp_a)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            pg_loss = - tf.reduce_mean(tf.minimum(self.adv * ratio, self.adv * ratio_clipped))
            
            #value loss
            vf_loss = tf.reduce_mean(tf.square(self.v - self.R))
            
            # Total loss
            loss = pg_loss + vf_loss

            optim = tf.train.AdamOptimizer(self.lr_a)
            grads = optim.compute_gradients(loss , var_list=self.params)
            if self.clip_norm:
                grads = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads]
            self.train_op = optim.apply_gradients(grads)
            self.train_ops = [self.train_op]


        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')

            self.summary = [
                tf.summary.scalar('loss/adv', tf.reduce_mean(self.adv)),
                tf.summary.scalar('loss/ratio', tf.reduce_mean(ratio)),
                tf.summary.scalar('loss/loss_actor', pg_loss),
                tf.summary.scalar('loss/loss_critic', vf_loss),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]

            # self.summary += [tf.summary.scalar('grads/' + v.name, tf.norm(g))
            #                 for g, v in grads_a if g is not None]
            # self.summary += [tf.summary.scalar('grads/' + v.name, tf.norm(g))
            #                 for g, v in grads_c if g is not None]

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(Config):
        lr_a = 0.005
        lr_c = 0.005
        batch_size = 32
        n_iterations = 100
        n_rollout_workers = 5
        train_epoches = 5
        log_every_iteration = 10
        ratio_clip_range = 0.2
        ratio_clip_decay = True

    def _generate_rollout(self, buffer):
        # generate one trajectory.
        ob = self.env.reset()
        done = False
        rewards = []
        episode_reward = 0.0
        obs = []
        actions = []

        while not done:
            a = self.act(ob)
            ob_next, r, done, info = self.env.step(a)
            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            episode_reward += r
            ob = ob_next

        # length of the episode.
        T = len(rewards)

        # compute the current log pi(a|s) and predicted v values.
        with self.sess.as_default():
            logp_a = self.logp_a.eval({self.a: np.array(actions), self.s: np.array(obs)})
            v_pred = self.v.eval({self.s: np.array(obs)})


        assert len(logp_a) == len(v_pred)


        mb_rewards = np.asarray(rewards, dtype=np.float32)
        v_pred = np.array(v_pred)
        last_value = v_pred[-1]

        # discount/bootstrap off value fn
        returns = np.zeros_like(mb_rewards)
        advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0
                nextvalues = v_pred[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - v_pred[t]
            advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        returns = advs + v_pred

        assert len(advs) == T

        # add into the memory buffer
        for i, (s, a, s_next, r, old_logp_a, R, adv) in enumerate(zip(
                obs, actions, np.array(obs[1:] + [ob_next]), rewards,
                np.squeeze(logp_a), returns, advs)):
            done = float(i == T - 1)
            buffer.add(buffer.tuple_class(s, a, s_next, r, done, old_logp_a, R, adv))

        return episode_reward, len(advs)

    def train(self, config: TrainConfig):
        BufferRecord = namedtuple('BufferRecord', ['s', 'a', 's_next', 'r', 'done',
                                                   'old_logp_actor', 'R', 'adv'])
        buffer = ReplayMemory(tuple_class=BufferRecord)

        reward_history = []
        reward_averaged = []
        step = 0
        total_rec = 0

        clip = config.ratio_clip_range
        if config.ratio_clip_decay:
            clip_delta = clip / config.n_iterations
        else:
            clip_delta = 0.0

        for n_iteration in range(config.n_iterations):

            # we should have multiple rollout_workers running in parallel.
            for _ in range(config.n_rollout_workers):
                episode_reward, n_rec = self._generate_rollout(buffer)
                # One trajectory is complete.
                reward_history.append(episode_reward)
                reward_averaged.append(np.mean(reward_history[-10:]))
                total_rec += n_rec

            # now let's train the model for some steps.
            for batch in buffer.loop(config.batch_size, epoch=config.train_epoches):
                _, summ_str = self.sess.run(
                    [self.train_ops, self.merged_summary], feed_dict={
                        self.lr_a: config.lr_a,
                        self.lr_c: config.lr_c,
                        self.clip_range: clip,
                        self.s: batch['s'],
                        self.a: batch['a'],
                        self.s_next: batch['s_next'],
                        self.r: batch['r'],
                        self.done: batch['done'],
                        self.old_logp_a: batch['old_logp_actor'],
                        self.R: batch['R'],
                        self.adv: batch['adv'],
                        self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,
                    })

                self.writer.add_summary(summ_str, step)
                step += 1

            clip = max(0.0, clip - clip_delta)

            if (reward_history and config.log_every_iteration and
                    n_iteration % config.log_every_iteration == 0):
                # Report the performance every `log_every_iteration` steps
                print("[iteration:{}/step:{}], best:{}, avg:{:.2f}, hist:{}, clip:{:.2f}; {} transitions.".format(
                    n_iteration, step, np.max(reward_history), np.mean(reward_history[-10:]),
                    list(map(lambda x: round(x, 2), reward_history[-5:])), clip, total_rec
                ))
                # self.save_checkpoint(step=step)

        self.save_checkpoint(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
