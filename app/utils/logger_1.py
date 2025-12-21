import logging
import os
from datetime import datetime

# 1. Define the directory where logs should stay
log_dir = os.path.join(os.getcwd(), "logs")

# 2. Create the 'logs' DIRECTORY if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# 3. Define the specific FILE name
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


if __name__=='__main__':
    logging.info("This is a test log message!!")