import logging
import os

def get_logger(log_path):# {{
  logger = logging.getLogger(__name__)
  
  if os.path.exists(os.path.join(log_path)):
    os.remove(os.path.join(log_path))
  file_hdler = logging.FileHandler(log_path)
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
  file_hdler.setFormatter(formatter)
  logger.addHandler(file_hdler)
  return logger# }}

  


