# -*- coding: utf-8 -*-

"""
    File name    :    grpc_server
    Date         :    10/05/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""

import time, grpc, numpy as np
from concurrent import futures
from grpc_utils_flatten import batch_data_pb2, batch_data_pb2_grpc


# 实现 proto 文件中定义的 GreeterServicer
class TransferBatchData(batch_data_pb2_grpc.TransferBatchDataServicer):
    # 实现 proto 文件中定义的 rpc 调用
    def Send(self, request, context):
        t1 = time.time()
        s = np.reshape(np.array(request.states, dtype=np.uint8), newshape=(5, 32, 84, 84, 4))
        print("Reshape states time:", time.time() - t1)
        a = np.reshape(np.array(request.actions, dtype=np.float32), newshape=(5, 32, 6))
        r = np.reshape(np.array(request.rewards, dtype=np.float32), newshape=(5, 32))
        print(s.shape, a.shape, r.shape)
        # print("ok" if (a == np.ones(shape=(5, 32, 6), dtype=np.float32)).all() else "data error")
        return batch_data_pb2.ReceiveReply(boolean=True)


def serve():
    # 启动 rpc 服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20), options=[
        ('grpc.max_send_message_length', 10 * 1024 * 1024),
        ('grpc.max_receive_message_length', 10 * 1024 * 1024)])

    batch_data_pb2_grpc.add_TransferBatchDataServicer_to_server(TransferBatchData(), server)
    server.add_insecure_port('127.0.0.1:50051')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
