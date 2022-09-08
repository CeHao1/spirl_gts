
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")

    # Folder settings
    parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory")

    return parser.parse_args()