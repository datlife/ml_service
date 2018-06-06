import re
import os
import csv
import numpy as np


def parse_label_map(label_map_path):
  """Parse label map file into a dictionary
  Args:
    label_map_path:

  Returns:
    a dictionary : key: obj_id value: obj-name
  """
  # Open image has different label maps:  {name: ... id: .... display_name: ...}
  # MSCOCO and PASCAL has: {id:... name: ...}
  is_oid = False
  if 'oid' in label_map_path.split('/')[-1]:
    is_oid = True
    parser = re.compile(r'name:\s+\"(?P<name>[/\w_-]+)\"\s+id:\s+(?P<id>[0-9]+)\s+display_name:\s+\"(?P<display_name>[\w_-]+)\"')
  else:
      parser = re.compile(r'id:[^\d]*(?P<id>[0-9]+)\s+name:[^\']*\'(?P<name>[\w_-]+)\'')
  # Read label_map file
  with open(label_map_path, 'r') as f:
    lines = f.read().splitlines()
    lines = ''.join(lines)

    # match group having the same language 
    result = parser.findall(lines)
    if is_oid:
      label_map_dict = {int(item[1]): item[2] for item in result}
    else:
      label_map_dict = {int(item[0]): item[1] for item in result}

    return label_map_dict


def parse_inputs(filename, label_dict):
  """Read input file and convert into inputs, labels for training

  Args:
    filename: text file - has content format as
        image_path, x1, x2, y1, y2, label1
        image_path, x1, x2, y1, y2, label2
    label_dict:  an encoding dictionary -
      mapping class names to indices

  Returns:
    inputs :  a list of all image paths to dataset
    labels :  a dictionary,
      key : image_path
      value: all objects in that image

  @TODO: when dataset is large, we should consider this method a generator
  """
  training_instances = dict()
  with open(filename, "r") as f:
    reader = csv.reader(f)
    for line in islice(reader, 1, None):
      if not line:
        continue  # Ignore empty line

      img_path = line[0]
      cls_name = line[-1]
      x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
      an_object = [y1, x1, y2, x2, label_dict[cls_name]]

      if img_path in training_instances:
        training_instances[img_path].append(an_object)
      else:
        training_instances[img_path] = [an_object]
  inputs = training_instances.keys()
  labels = {k: np.stack(v).flatten() for k, v in training_instances.items()}
  return inputs, labels