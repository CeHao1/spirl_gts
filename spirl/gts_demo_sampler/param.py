
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    parser.add_argument("--ip_address", default=None, help="ip address of the gts")

    # Folder settings
    parser.add_argument("--prefix", default=None, help="experiment prefix, if given creates subfolder in experiment directory")

    return parser.parse_args()