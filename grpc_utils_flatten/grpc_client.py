# -*- coding: utf-8 -*-

"""
    File name    :    grpc_utils_flatten
    Date         :    10/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

import grpc, numpy as np
from grpc_utils_flatten import batch_data_pb2, batch_data_pb2_grpc


def run():
    shared_states = np.zeros(shape=(5, 32, 84, 84, 4), dtype=np.uint8).flatten()
    shared_actions = np.ones(shape=(5, 32, 6), dtype=np.float32).flatten()
    shared_rewards = np.zeros(shape=(5, 32,), dtype=np.float32).flatten()


    # 连接 rpc 服务器

    channel = grpc.insecure_channel('127.0.0.1:50051')
    # 调用 rpc 服务
    stub = batch_data_pb2_grpc.TransferBatchDataStub(channel)
    for _ in range(6):
        response = stub.Send(batch_data_pb2.BatchData(states=shared_states, actions=shared_actions, rewards=shared_rewards))
        print("Transfer client received: " + str(response.boolean))


if __name__ == '__main__':
    run()
