"""Tensorflow Serving libraries
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import signal
import subprocess
import time

import numpy as np
import tensorflow as tf
# TensorFlow serving python API to send messages to server
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

UNIX_COMMAND = "tensorflow_model_server --port={} --model_name={} --model_base_path={} \
                --per_process_gpu_memory_fraction={}"


class DetectionClient(object):
    """This object is responsible for:

    * Create a detection request
    * Send the request to the server
    * Interpret the result and send back to whoever calls it
    """

    def __init__(self, server, model, label_dict, verbose=False):
        self.host, self.port = server.split(':')
        self.model = model
        self.label_dict = label_dict
        self.verbose = verbose

        channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def predict(self, image, img_dtype=tf.uint8, timeout=20.0):
        request = predict_pb2.PredictRequest()

        start = time.time()
        image = np.expand_dims(image, axis=0)

        request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(image,
                                                               dtype=img_dtype))
        request.model_spec.name = self.model
        request.model_spec.signature_name = 'predict_images'

        pred = time.time()
        result = self.stub.Predict(request, timeout)  # 20 secs timeout

        if self.model == 'detector':
            num_detections = -1
        else:
            num_detections = int(result.outputs['num_detections'].float_val[0])

        classes = result.outputs['detection_classes'].float_val[:num_detections]
        scores = result.outputs['detection_scores'].float_val[:num_detections]
        boxes = result.outputs['detection_boxes'].float_val[:num_detections * 4]
        classes = [self.label_dict[int(idx)] if idx in self.label_dict.keys() else -1 for idx in classes ]
        boxes = [boxes[i:i + 4] for i in range(0, len(boxes), 4)]
        if self.verbose:
            print("Number of detections: %s" % len(classes))
            print("Server Prediction in {:.3f} ms || Total {:.3} ms".format(1000*(time.time() - pred),
                                                                            1000*(time.time() - start)))
        return boxes, classes, scores


class DetectionServer(object):
    """Manage detection Server for interference

    This object will manage turning on/off server
    """

    def __init__(self, model, model_path, port=9000, per_process_gpu_memory_fraction=0.0):
        """
        Args:
          model: name of detection model -
            should match with the directory model
          model_path: path to the directory containing frozen model
          port: an int - port to create detection server
        """
        self.server = None
        self.running = False
        self.model_path = model_path
        self.model = model
        self.port = port
        self.gpu_mem = per_process_gpu_memory_fraction

    def is_running(self):
        return self.running

    def start(self):
        return self._callback(command='start')

    def stop(self):
        return self._callback(command='stop')

    def _callback(self, command):
        if command == 'start':
            if not self.running:
                print("Serving Server is launching ... ")
                self.server = subprocess.Popen(UNIX_COMMAND.format(
                    self.port,
                    self.model,
                    self.model_path,
                    self.gpu_mem),
                    stdin=subprocess.PIPE, shell=True)
                print("Serving Server is started at PID %s\n" % self.server.pid)
                self.running = True
            else:
                print("Serving Server has been activated already..\n")

        if command == 'stop':
            if self.running:
                self.running = True
                self._turn_off_server()
                print("Serving Server is off now\n")
            else:
                print("Serving Server is not activated yet..\n")

        return self

    def _turn_off_server(self):
        ps_command = subprocess.Popen("kill -9 $(lsof -t -i:%d -sTCP:LISTEN)" % self.port,
                                      shell=True,
                                      stdout=subprocess.PIPE)

        ps_output = ps_command.stdout.read()
        return_code = ps_command.wait()
        for pid_str in ps_output.split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGINT)
        self.server.terminate()
