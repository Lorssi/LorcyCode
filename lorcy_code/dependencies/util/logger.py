import os
import sys
import time
import logging
from datetime import datetime, timezone
import pytz


def init_logger(log_file_save_path: str = '../../log/morning') -> logging.Logger:
    """
    初始化日志对象
    :param log_file_save_path: 日志文件保存路径
    :return: 日志对象
    """
    timezone = pytz.timezone('Asia/Shanghai')
    log_time = datetime.now(timezone).strftime('%Y-%m-%d_%H-%M-%S')
    log_file_save_path = os.path.join(log_file_save_path, log_time + '.log')

    if not os.path.exists(os.path.dirname(log_file_save_path)):
        os.makedirs(os.path.dirname(log_file_save_path))
    formater = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(log_file_save_path, encoding='utf-8')
    file_handler.setFormatter(formater)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    logger.info('log file saved to: {}'.format(log_file_save_path))
    return logger
