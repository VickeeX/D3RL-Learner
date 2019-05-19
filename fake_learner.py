# -*- coding: utf-8 -*-

"""
    File name    :    learner
    Date         :    14/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

import multiprocessing as mp, numpy as np, time
from zmq_server import zmq_server_run


class FakeLearner:
    def __init__(self):
        self.shared = []

    def train(self):
        """ train"""
        # time.sleep(5)
        for i in range(6):
            dt = get_batch()
            print(dt.__sizeof__())
            print("get batch", i, "ok")

            # print(len(data), data[0][1].shape, data[0][2].shape, data[0][3].shape)
            # print("ok" if data[0][3] == np.ones(shape=(32,), dtype=np.float32) else "data wrong")


def put_batch(data):
    queue.put(data)
    print("put ok")
    # print(self.queue.qsize())


def get_batch():
    return queue.get()


def fake_server():
    shared_states = np.zeros(shape=(5, 32, 84, 84, 4), dtype=np.uint8)
    shared_actions = np.zeros(shape=(5, 32, 6), dtype=np.float32)
    shared_rewards = np.zeros(shape=(5, 32,), dtype=np.float32)
    data = list(enumerate(zip(shared_states, shared_actions, shared_rewards)))
    print(data[0].__sizeof__(), data[1].__sizeof__())
    put_batch(data[0])
    put_batch(data[1])


if __name__ == '__main__':
    queue = mp.Queue(maxsize=10240)
    learner = FakeLearner()
    # mp.Process(target=fake_server).run()
    mp.Process(target=zmq_server_run).run()
    learner.train()
