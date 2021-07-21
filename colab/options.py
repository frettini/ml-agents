import json

def get_options():
    with open('C:/Users/nicol/Work/Master/dissertation/ml-agents/colab/options.json') as f:
        options = json.load(f)
    return options
