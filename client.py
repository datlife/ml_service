
import cv2 
import yaml
import numpy as np

from ml_service.utils.painter import draw_boxes
from ml_service.utils.parser import parse_label_map

from ml_service.object_detection.ObjectDetection import ObjectDetection

def main():
  # ############
  # Parse Config
  # ############
  with open('config.yml', 'r') as stream:
    config = yaml.load(stream)
  model_name = config['model_name']
  inference = config['inference']
  label_dict = parse_label_map(config['label_map'])
  
  img = cv2.imread('camera0.jpg')
  h, w, _ = img.shape

  # Initialize Object Detection Client
  object_detector = ObjectDetection(
      inference['host'], 
      inference['port'],
      model_name, 
      label_dict, 
      verbose=True)

  print('Detecting objects...')
  bboxes, classes, scores = object_detector.predict(img, img_dtype=np.uint8, timeout=60)

  filtered_outputs = [(box, idx, score) for box, idx, score in zip(bboxes, classes, scores)
                      if score > inference['score_threshold']]
  if zip(*filtered_outputs):
    boxes, classes, scores = zip(*filtered_outputs)
    boxes = [box * np.array([h, w, h, w]) for box in boxes]
    img = draw_boxes(img, boxes, classes, scores)

  cv2.imwrite('output.jpg', img)
  print('Done!')


if __name__ =='__main__':
  main()