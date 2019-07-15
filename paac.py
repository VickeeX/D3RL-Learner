import time, logging, zmq, multiprocessing as mp, requests
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
from zmq_serialize import SerializingContext


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, args):
        super(PAACLearner, self).__init__(network_creator, args)
        # self.workers = args.emulator_workers
        self.queue = mp.Queue(maxsize=102400)
        self.zmq_server_proc = mp.Process(target=self.zmq_server_run)
        self.cpu_num = args.cpu_num

    def zmq_server_run(self):
        ctx = SerializingContext()
        rep = ctx.socket(zmq.REP)
        rep.bind("tcp://127.0.0.1:6666")
        count = 0
        stop_worker = self.cpu_num
        batch = self.max_local_steps * self.emulator_counts
        while True:
            data = rep.recv_zipped_pickle()
            # print("Checking zipped pickle...")
            count += batch
            if count > self.max_global_steps:
                rep.send_string("stop")
                stop_worker -= 1
                if stop_worker <= 0:
                    break
            else:
                self.put_batch(data)
                rep.send_string("received data.")
        rep.close()

    def put_batch(self, data):
        self.queue.put(data)

    def get_batch(self):
        return self.queue.get()

    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi = session.run(
                [network.output_layer_v,
                 network.output_layer_pi],
                feed_dict={network.input_ph: states})

        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def __choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """
        self.zmq_server_proc.start()
        self.global_step = self.init_network()

        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0

        global_step_start = self.global_step

        total_rewards = []

        # state, reward, episode_over, action
        # variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
        #              (np.zeros(self.emulator_counts, dtype=np.float32)),
        #              (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
        #              (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        # self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        # self.runners.start()
        # shared_states, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        summaries_op = tf.summary.merge_all()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        # rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        # states = np.zeros([self.max_local_steps, self.emulator_counts] + [84, 84, 4], dtype=np.uint8)
        # actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        # values = np.zeros((self.max_local_steps, self.emulator_counts))
        # episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            data = self.get_batch()
            states, rewards, episodes_over_masks, actions, values = data[0], data[1], data[2], data[3], data[4]
            # print("Get batch data:", self.global_step, "from queue")
            self.global_step += max_local_steps * self.emulator_counts
            # for t in range(max_local_steps):
            #     next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(shared_states)
            #     actions_sum += next_actions
            #     for z in range(next_actions.shape[0]):
            #         shared_actions[z] = next_actions[z]
            #
            #     actions[t] = next_actions
            #     values[t] = readouts_v_t
            #     states[t] = shared_states
            #
            #     # Start updating all environments with next_actions
            #     self.runners.update_environments()
            #     self.runners.wait_updated()
            #     # Done updating all environments, have new states, rewards and is_over
            #     episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)
            #
            #     for e, (actual_reward, episode_over) in enumerate(zip(rewards[-1], episodes_over_masks[-1])):
            #         total_episode_rewards[e] += actual_reward
            #         actual_reward = self.rescale_reward(actual_reward)
            #         rewards[t, e] = actual_reward
            #
            #         emulator_steps[e] += 1
            #         self.global_step += 1
            #         if episode_over:
            #             total_rewards.append(total_episode_rewards[e])
            #             episode_summary = tf.Summary(value=[
            #                 tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e]),
            #                 tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e]),
            #             ])
            #             self.summary_writer.add_summary(episode_summary, self.global_step)
            #             self.summary_writer.flush()
            #             total_episode_rewards[e] = 0
            #             emulator_steps[e] = 0
            #             actions_sum[e] = np.zeros(self.num_actions)

            nest_state_value = self.session.run(
                    self.network.output_layer_v,
                    feed_dict={self.network.input_ph: states[-1]})  # shared_states

            estimated_return = np.copy(nest_state_value)

            for t in reversed(range(max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                y_batch[t] = np.copy(estimated_return)
                adv_batch[t] = estimated_return - values[t]

            flat_states = states[:-1].reshape([self.max_local_steps * self.emulator_counts] + [84, 84, 4])
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)

            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}

            _, summaries = self.session.run(
                    [self.train_step, summaries_op],
                    feed_dict=feed_dict)

            self.summary_writer.add_summary(summaries, self.global_step)
            self.summary_writer.flush()

            counter += 1

            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                logging.info("Ran {} steps, at {} steps/s.".format(global_steps,
                                                                   self.max_local_steps * self.emulator_counts / (
                                                                   curr_time - loop_start_time)))
            self.save_vars()

            # transfer network checkpoint to workers
            if counter % 16 == 0:
                self.last_saving_step = self.global_step
                self.network_saver.save(self.session, self.network_checkpoint_folder, global_step=self.last_saving_step)
                ckpt_path = tf.train.latest_checkpoint(self.network_checkpoint_folder)
                file = [('files', open(ckpt_path + '.meta', 'rb')),
                        ('files', open(ckpt_path + '.index', 'rb')),
                        ('files', open(ckpt_path + '.data-00000-of-00001', 'rb'))]

                Process(target=self.post_network_ckpt,
                        kwargs={"ip": "10.0.201.96", "path": ckpt_path, "file": file}).start()
                Process(target=self.post_network_ckpt,
                        kwargs={"ip": "10.0.201.98", "path": ckpt_path, "file": file}).start()

        self.cleanup()

    def post_network_ckpt(self, ckpt_path, ip, file):

        r = requests.post('http://' + ip + ':6668/d3rl/network', files=file)
        # print(r.text)
        logging.info("Post network ckpt: {} to {}.".format(ckpt_path, ip))

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.zmq_server_proc.terminate()
        # self.runners.stop()
