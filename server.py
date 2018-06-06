"""Object detection Server"""
import yaml
from ml_service.TFServingServer import TFServingServer

def main():
  # Parse Config
  with open('config.yml', 'r') as stream:
    config = yaml.load(stream)
  model_name = config['model_name']
  model_path = config['model_path']

  # Init Server
  tfserving_server = TFServingServer(
      port=9000, 
      model_name=model_name, 
      model_path=model_path)
  tfserving_server.start()

  # Wait for server to start
  if tfserving_server.is_running():
    try:
      while tfserving_server.is_running():
        continue
    except KeyboardInterrupt as e:
      print("\nWaiting for last predictions before turning off...")
      tfserving_server.stop()


if __name__ == "__main__":
    main()
