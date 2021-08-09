from mlagents.torch_utils import torch, default_device
from options import get_options
from skeleton_temporal import SkeletonConv, SkeletonPool, SkeletonUnpool, SkeletonLinear, find_neighbor
import numpy as np



class Encoder(torch.nn.Module):
    def __init__(self, topology):
        super(Encoder, self).__init__()
        options = get_options()

        self.topologies = [topology]

        self.channel_base = [options['channel_base']]
        
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = torch.nn.ModuleList()
        self.convs = []
        num_layers = options['num_layers']

        # initialize convolution kernel
        kernel_size = options['kernel_size']
        padding = (kernel_size - 1) // 2
        bias = True
        add_offset = True # default concat

        # construct list of number of channels, multiply by 2 at every layer 
        for i in range(num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        # Construct the layers and their operators
        for i in range(num_layers):
            seq = [] 
            neighbor_list = find_neighbor(self.topologies[i], options['num_neighbours']) #default 2

            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]

            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            # for _ in range(args.extra_conv): #default 0
            #     seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
            #                             joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
            #                             padding=padding, padding_mode=args.padding_mode, bias=bias))

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))

            self.convs.append(seq[-1])

            last_pool = True if i == num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode='mean',
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(torch.nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(torch.nn.Sequential(*seq))

            self.topologies.append(pool.new_edges) # add the new topologies created by the pooling operator
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        """
        The input size is:
        [batch_size, window_size, n_joints*channel_size]
        """
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        # if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            # input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)
        for i, layer in enumerate(self.layers):
            # if offset is not None: #default concat
            self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class Decoder(torch.nn.Module):
    def __init__(self, enc: Encoder):
        super(Decoder, self).__init__()

        options = get_options()

        self.layers = torch.nn.ModuleList()
        self.unpools = torch.nn.ModuleList()

        self.enc = enc # needs a reference to the encoder for the unpooling operation
        self.convs = []
        num_layers = options['num_layers']

        kernel_size = options['kernel_size']
        padding = (kernel_size - 1) // 2
        add_offset = True

        # test dropout
        self.drop_percentage = options['drop_percentage']
        if self.drop_percentage is not None:
            self.drop = torch.nn.Dropout(p=self.drop_percentage)
        

        for i in range(num_layers):
            seq = []
            in_channels = enc.channel_list[num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], options['num_neighbours'])

            if i != 0 and i != num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[num_layers - i - 1], in_channels // len(neighbor_list)))

            # add the upsampling, unpooling and squeleton to the layers
            seq.append(torch.nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            seq.append(self.unpools[-1])

            # for _ in range(args.extra_conv):
            #     seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
            #                             joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size,
            #                             stride=1,
            #                             padding=padding, padding_mode=args.padding_mode, bias=bias))
            
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])

            if i != num_layers - 1: seq.append(torch.nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(torch.nn.Sequential(*seq))

    def forward(self, input, offset=None):

        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])

            # dropout: turned of for test
            if self.drop_percentage is not None and input.requires_grad is False:
                 input = self.drop(input)

            input = layer(input)

        # throw the padded rwo for global position
        # if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
        #     input = input[:, :-1, :]

        return input

# eoncoder for static part, i.e. offset part
class StaticEncoder(torch.nn.Module):
    def __init__(self, edges):
        
        super(StaticEncoder, self).__init__()

        options = get_options()
        
        self.layers = torch.nn.ModuleList()
        activation = torch.nn.LeakyReLU(negative_slope=0.2)
        channels = 3
        num_layers = options['num_layers']

        for i in range(num_layers):
            neighbor_list = find_neighbor(edges, options['num_neighbours'])
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))

            if i < num_layers - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges

            seq.append(activation)
            channels *= 2
            self.layers.append(torch.nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input: torch.Tensor):
        # record every output from every layer since we are going to use it later
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))
        return output


class AE(torch.nn.Module):
    def __init__(self, topology):
        super(AE, self).__init__()
        self.enc = Encoder(topology)
        self.dec = Decoder(self.enc)

    def forward(self, input, offset=None):

        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, result

class CrossStructuralAE(torch.nn.Module):

    def __init__(self, skdata_data, skdata_sim):
        """
        AutoEncoder with different input and output structures
        """

        super(CrossStructuralAE, self).__init__()
        
        self.device = default_device()
        self.options = get_options()

        self.encoder_sim = Encoder(skdata_sim.edges)
        # use the initialization of encoder_data to get the correct pooling lists
        # its forward is not used
        encoder_data = Encoder(skdata_data.edges) 
        self.decoder_data = Decoder(encoder_data)

        self.static_encoder_sim = StaticEncoder(skdata_sim.edges).to(self.device)
        self.static_encoder_data = StaticEncoder(skdata_data.edges).to(self.device)

    def forward(self, input_motion, offset_data, offset_sim):

        # first get the offsets from the static encoder 
        # deep_offsets = static_encoder(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
        deep_offsets_sim = self.static_encoder_sim(self.skdata_sim.offsets.reshape(1, self.skdata_sim.offsets.shape[0], -1))
        deep_offsets_data = self.static_encoder_data(self.skdata_data.offsets.reshape(1, self.skdata_data.offsets.shape[0], -1))

        latent = self.encoder_sim(input_motion, deep_offsets_sim)
        res = self.decoder_data(latent, deep_offsets_data)

        return latent, res
