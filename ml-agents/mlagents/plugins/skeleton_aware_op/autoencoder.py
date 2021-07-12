from mlagents.torch_utils import torch, default_device
from skeleton import SkeletonConv, SkeletonPool, SkeletonUnpool, SkeletonLinear, find_neighbor
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, edges):
        """
        edges - n by 2 array of joint indices that represent an edge between the two
        """
        super(Encoder, self).__init__()

        self.topologies = [edges]
        # if args.rotation == 'euler_angle': self.channel_base = [3]
        
        self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(edges) + 1]
        self.pooling_list = []
        self.layers = torch.nn.ModuleList()
        self.convs = []
        self.num_layers = 2     # TODO: remove hard code
        self.skeleton_dist = 2  # TODO: remove hard code

        # bias is a learnable paramater added to the convolution
        bias = True
        add_offset = True 

        for i in range(self.num_layers): #default 2
            # multiply the number of channels by two for every layer [4,8,16,...]
            self.channel_base.append(self.channel_base[-1] * 2)

        # Construct the layers and their operators -> skeletonConv, skeletonPooling
        for i in range(self.num_layers):

            seq = []
            # construct list of neighbours of distance skeleton_dist
            neighbor_list = find_neighbor(self.topologies[i], self.skeleton_dist) #default 2
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            # for _ in range(args.extra_conv): #default 0
            #     seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
            #                             joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
            #                             padding=padding, padding_mode=args.padding_mode, bias=bias))

            # conv_layer = SkeletonConv(neighbour_list, 23*4, 23*8, 23, bias=True, add_offset=False, in_offset_channel=0)


            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))

            self.convs.append(seq[-1])

            # check if this is the last pooling
            last_pool = True if i == self.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode="mean",
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(torch.nn.LeakyReLU(negative_slope=0.2))
            
            # add layer to the sequence of layers
            self.layers.append(torch.nn.Sequential(*seq))

            # store the edges and pooling list so they can be used in the decoder
            self.topologies.append(pool.new_edges) 
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)

            # not used?
            # if i == self.num_layers - 1:
            #     self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        """
        input - [batch_size, num_joints*4] array containing the quaternion rotation of each joint.
        """
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        # will need to change the data representation 
        # if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
        # input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            # if self.args.skeleton_info == 'concat' and offset is not None: #default concat
            self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class Decoder(torch.nn.Module):
    def __init__(self, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.unpools = torch.nn.ModuleList()
        self.enc = enc # needs a reference to the encoder for the unpooling operation
        self.convs = []
        self.num_layers = 2
        self.skeleton_dist = 2

        add_offset = True
        
        for i in range(self.num_layers):
            seq = []
            in_channels = enc.channel_list[self.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[self.num_layers - i - 1], self.skeleton_dist)

            if i != 0 and i != self.num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[self.num_layers - i - 1], in_channels // len(neighbor_list)))

            # add the upsampling, unpooling and squeleton to the layers
            
            #time upsampling is not needed here
            # seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            
            seq.append(self.unpools[-1])
            
            # for _ in range(args.extra_conv):
            #     seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
            #                             joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size,
            #                             stride=1,
            #                             padding=padding, padding_mode=args.padding_mode, bias=bias))
            
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[self.num_layers - i - 1], bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[self.num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])

            if i != self.num_layers - 1: seq.append(torch.nn.LeakyReLU(negative_slope=0.2))
            # else: seq.append(torch.nn.Sigmoid())

            self.layers.append(torch.nn.Sequential(*seq))


    def forward(self, input, offset=None):

        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)

        # # throw the padded rwo for global position
        # if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
        #     input = input[:, :-1, :]
        # input = input[:,:-1]

        return input


class StaticEncoder(torch.nn.Module):
    def __init__(self, edges):

        torch.device(default_device())

        super(StaticEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        activation = torch.nn.LeakyReLU(negative_slope=0.2)
        channels = 3
        self.num_layers = 2     # TODO: remove hard code
        self.skeleton_dist = 2  # TODO: remove hard code

        # construct encoder layers
        for i in range(self.num_layers):

            neighbor_list = find_neighbor(edges, self.skeleton_dist)
            seq = []
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))

            if i < self.num_layers - 1:
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


class AutoEncoder(torch.nn.Module):
    def __init__(self, edges):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(edges)
        self.dec = Decoder(self.enc)

    def forward(self, input, offset=None):

        latent = self.enc(input, offset)
        result = self.dec(latent, offset)
        return latent, torch.sigmoid(result)
