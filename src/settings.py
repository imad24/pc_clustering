# -*- coding: utf-8 -*-
"""This file contains the global settings of the package
- Make sure to add the installation folder path to PYTHONPATH, this way this file could be included in all script files
"""
import os
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

def init():
    global root_dir
    global subfolder
    global PREFIX
    global raw_path
    global interim_path
    global processed_path
    global reports_path
    global models_path
    
    global row_headers
    global n_row_headers

    global options

    root_dir = os.getenv("APPPATH")
    subfolder = os.getenv("SUBFOLDER")
    PREFIX = os.getenv("PREFIX")
    raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
    interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
    processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

    reports_path = os.path.join(root_dir,"reports\\",subfolder)
    models_path = os.path.join(root_dir,"models\\",subfolder)
    row_headers = ["Product"]
    n_row_headers = len(row_headers)

    options = {
        "windows_size" : 2,
        "smoothing_method" : "average",
        "range" : 16,
        "offset" : 1,
        "init_method" : "PCA",
        "best_k" : 4
    }

init()