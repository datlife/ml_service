"""Tensorflow Serving Object Detection Server
"""
import os
import signal
import time
import subprocess

UNIX_COMMAND = \
"tensorflow_model_server --port={} --model_name={} --model_base_path={} --per_process_gpu_memory_fraction={}"

class TFServingServer(object):
  """Manage TF Serving Server for inference.
  This object will manage turning on/off server
  """
  def __init__(self, port, model_name, model_path, per_process_gpu_memory_fraction=0.0):
    """
    Args:
      model_name: name of detection model -
        should match with the directory model
      model_path: path to the directory containing frozen model
      port: an int - port to create detection server
    """
    self.port = port
    self.server = None
    self.running = False
    
    self.model_path = model_path
    self.model_name = model_name
    self.gpu_mem = per_process_gpu_memory_fraction

  def is_running(self):
    return self.running

  def start(self):
    if not self.running:
      print("Serving Server is launching ... ")
      self.server = subprocess.Popen(UNIX_COMMAND.format(
          self.port,
          self.model_name,
          self.model_path,
          self.gpu_mem),
          stdin=subprocess.PIPE, shell=True)
      print("Serving Server is started at PID %s\n" % self.server.pid)
      self.running = True
    else:
      print("Serving Server has been activated already..\n")

  def stop(self):
    if self.running:
      self.running = True
      self._turn_off_server()
      print("Serving Server is off now\n")
    else:
      print("Serving Server is not activated yet..\n")

  def _turn_off_server(self):
    ps_command = subprocess.Popen(
        "kill -9 $(lsof -t -i:%d -sTCP:LISTEN)" % self.port,
        shell=True,
        stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    for pid_str in ps_output.split("\n")[:-1]:
      os.kill(int(pid_str), signal.SIGINT)
    self.server.terminate()
