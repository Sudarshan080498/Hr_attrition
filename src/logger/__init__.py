import logging
import os,sys
from datetime import datetime

Log_dir = "logs"
Log_dir= os.path.join(os.getcwd(), Log_dir)


os.makedirs(Log_dir, exist_ok=True)


Current_time_stamp= f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
file_name = f"log_{Current_time_stamp}.log"



log_file_path = os.path.join(Log_dir, file_name)


logging.basicConfig(filename= log_file_path,
                    filemode="w",
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)