# -*- coding: utf-8 -*-

"""
    File name    :    ZMQ_utils
    Date         :    08/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

"""A Socket subclass that adds some serialization methods."""

import zlib, zmq, pickle
from fake_learner import put_batch


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods

    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.

    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """

    def send_zipped_pickle(self, obj, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        print('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags=flags)

    def recv_zipped_pickle(self, flags=0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

        # def send_array(self, A, flags=0, copy=True, track=False):
        #     """send a numpy array with metadata"""
        #     md = dict(
        #             dtype=str(A.dtype),
        #             shape=A.shape,
        #     )
        #     self.send_json(md, flags | zmq.SNDMORE)
        #     return self.send(A, flags, copy=copy, track=track)

        # def recv_array(self, flags=0, copy=True, track=False):
        #     """recv a numpy array"""
        #     md = self.recv_json(flags=flags)
        #     msg = self.recv(flags=flags, copy=copy, track=track)
        #     A = np.frombuffer(msg, dtype=np.float32)
        #     return A.reshape((32, 6))


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


def zmq_server_run():
    # shared_states = np.zeros(shape=(5, 32, 84, 84, 4), dtype=np.uint8)
    # shared_actions = np.zeros(shape=(5, 32, 6), dtype=np.float32)
    # shared_rewards = np.zeros(shape=(5, 32,), dtype=np.float32)
    # data = []
    # for e, (s, a, r) in enumerate(zip(shared_states, shared_actions, shared_rewards)):
    #     data.append((e, s, a, r))

    ctx = SerializingContext()
    rep = ctx.socket(zmq.REP)
    rep.bind("tcp://127.0.0.1:6666")

    while True:
        data = rep.recv_zipped_pickle()
        put_batch(data)
        print("Checking zipped pickle...")
        # print("Okay" if (shared_rewards[2] == B[2][3]).all() else "Failed")
        rep.send_string("received data.")
        print("ok.")

        # rep.close()
        # rep.send_array(A, copy=False)

#
# if __name__ == '__main__':
#     zmq_server_run()
