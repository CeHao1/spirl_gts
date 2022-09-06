import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")

    return parser.parse_args()