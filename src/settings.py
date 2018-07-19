# -*- coding: utf-8 -*-
"""This file contains the global settings of the package
- Make sure to add the installation folder path to PYTHONPATH, this way this file could be included in all script files
"""
import os
import logging
import json
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())


def init():
    global root_dir
    global src_dir
    global subfolder
    global PREFIX
    global raw_path
    global interim_path
    global processed_path
    global reports_path
    global models_path
    # global variables
    global row_headers
    global n_row_headers

    global options

    # global paths
    root_dir = os.getenv("APPPATH")
    src_dir = os.path.join(root_dir,'src')
    subfolder = os.getenv("SUBFOLDER")
    PREFIX = os.getenv("PREFIX")

    # subfolders paths
    raw_path = os.path.join(root_dir, "data\\raw\\", subfolder)
    interim_path = os.path.join(root_dir, "data\\interim\\", subfolder)
    processed_path = os.path.join(root_dir, "data\\processed\\", subfolder)
    reports_path = os.path.join(root_dir, "reports\\", subfolder)
    models_path = os.path.join(root_dir, "models\\", subfolder)

    # variables
    row_headers = ["Product"]
    n_row_headers = len(row_headers)

    # load global settings from config file
    cfg_file = os.path.join(src_dir,'config.json')
    with open(cfg_file) as cfg:
        options = json.load(cfg)


def get_logger(name):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_fmt)

    logger = logging.getLogger(name)

    # clean any set logger
    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
     
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(src_dir, "error.log"),"a", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(src_dir, "app.log"),"a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

init()
