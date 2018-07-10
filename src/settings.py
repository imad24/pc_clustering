# -*- coding: utf-8 -*-
import sys 
import os

# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

def init():
    global subfolder
    global PREFIX
    global raw_path
    global interim_path
    global processed_path

    global reports_path
    global models_path
    global row_headers
    global n_row_headers


    subfolder = os.getenv("SUBFOLDER")
    PREFIX = os.getenv("PREFIX")
    raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
    interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
    processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

    reports_path = os.path.join(root_dir,"reports\\",subfolder)
    models_path = os.path.join(root_dir,"models\\",subfolder)
    row_headers = ["Product"]
    n_row_headers = len(row_headers)

init()