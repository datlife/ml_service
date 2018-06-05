import os
import sys
import time
import re
import yaml

import cv2 
import numpy as np

from ml_service.utils.painter import draw_boxes
from ml_service.utils.tfserving import DetectionClient
# from ml_service.utils.parser import parse_label_map

def main():

  # ############
  # Parse Config
  # ############
  with open('config.yml', 'r') as stream:
    config = yaml.load(stream)

  model_name = config['model_name']
  inference = config['inference']
  server     = inference['server']
  label_dict = parse_label_map(config['label_map'])
  print(label_dict)
  
  sample = cv2.imread('camera0.jpg')
  h,w, _ = sample.shape

  print('Detecting objects...')
  object_detector = DetectionClient(server, model_name, label_dict, verbose=True)

  for i in range(5):
    boxes, classes, scores = object_detector.predict(sample, img_dtype=np.uint8, timeout=60)

    filtered_outputs = [(box, idx, score) for box, idx, score in zip(boxes, classes, scores)
                        if score > 0.2]
    print(filtered_outputs)

  print('Done!')


def parse_label_map(label_map_path):
  """Parse label map file into a dictionary """
  # match any group having language of {name:[name] id:[number] display_name:'name'}
  parser = re.compile(
    r'name:\s+\"(?P<name>[/\w_-]+)\"\s+id:\s+(?P<id>[0-9]+)\s+display_name:\s+\"(?P<display_name>[\w_-]+)\"')

  with open(label_map_path, 'r') as f:
    lines = f.read().splitlines()
    lines = ''.join(lines)
    # a tuple (id, name)
    result = parser.findall(lines)
    label_map_dict = {int(item[1]): item[2] for item in result}
    return label_map_dict

if __name__ =='__main__':
  main()