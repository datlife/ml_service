"""Object detection Server
"""
import os
import sys
import time
import yaml
import numpy as np
from ml_service.utils.tfserving import DetectionServer

def main():

  # ############
  # Parse Config
  # ############
  with open('config.yml', 'r') as stream:
    config = yaml.load(stream)
  inference = config['inference']

  model_name = config['model_name']
  model_path = os.path.join(sys.path[0], 'ml_service', 'object_detection', model_name)

  # #####################
  # Init Detection Server
  # #####################
  object_detection_server = DetectionServer(model=model_name, model_path=model_path)
  object_detection_server.start()
  time.sleep(5.0)

  # Wait for server to start
  if object_detection_server.is_running():
    print("Initialized TF Serving at {} with model {}".format(
        inference['server'], model_name))

    data = None
    print("Listing for object detection requests.... ")
    try:
      while object_detection_server.is_running():
        if data is None:
          time.sleep(0.5)
          continue
    # Stop server
    except KeyboardInterrupt as e:
      print("\nWaiting for last predictions before turning off...")
      time.sleep(5.0)
      object_detection_server.stop()


if __name__ == "__main__":
    main()
