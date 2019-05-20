# -*- coding: utf-8 -*-

"""
    File name    :    learner
    Date         :    14/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

from zmq_serialize import SerializingContext
import multiprocessing as mp, zmq


class FakeLearner:
    def __init__(self):
        self.shared = []
        self.queue = mp.Queue(maxsize=10240)

    def zmq_server_run(self):
        ctx = SerializingContext()
        rep = ctx.socket(zmq.REP)
        rep.bind("tcp://127.0.0.1:6666")

        # while True:
        for i in range(6):
            data = rep.recv_zipped_pickle()
            self.put_batch(data)
            print("Checking zipped pickle...")
            # print("Okay" if (shared_rewards[2] == B[2][3]).all() else "Failed")
            rep.send_string("received data.")
            print("ok.")

    def put_batch(self, data):
        self.queue.put(data)
        print("put ok")
        # print(self.queue.qsize())

    def get_batch(self):
        return self.queue.get()

    def train(self):
        """ train"""
        mp.Process(target=self.zmq_server_run).start()

        for i in range(6):
            dt = self.get_batch()
            print(dt.__sizeof__())
            print("get batch", i, "ok")

            # print(len(data), data[0][1].shape, data[0][2].shape, data[0][3].shape)
            # print("ok" if data[0][3] == np.ones(shape=(32,), dtype=np.float32) else "data wrong")
