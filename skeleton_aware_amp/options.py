import json
import os

def get_options(option_file = 'options.json'):
    script_dir = os.path.dirname(__file__)
    with open(script_dir + '/' + option_file) as f:
        options = json.load(f)
    return options
