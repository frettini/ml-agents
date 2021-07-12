import json

options = {'kernel_size'        : 5, # convolutional kernel size 
           'channel_base'       : 4, # number of features per joints
           'num_layers'         : 2, # number of layers in the auto-encoder
           'num_neighbours'     : 2, # distance of neighbours to consider during convolution
           'window_size'        : 16}# size of the window motion analyzed

def get_options():
    with open('./options.json') as f:
        options = json.load(f)
    return options




