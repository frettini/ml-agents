import skeleton
from skeleton import SkeletonConv, SkeletonPool
from mlagents.torch_utils import torch, default_device


class Discriminator(torch.nn.Module):
    def __init__(self, topology):

        super(Discriminator, self).__init__()

        self.topologies = [topology]
        self.channel_base = [3] # work on position?
        self.channel_list = []
        self.joint_num = [len(topology) + 1] # edges + 1
        self.pooling_list = []
        self.layers = torch.nn.ModuleList()
        self.num_layers = 2
        self.skeleton_dist = 2
        self.patch_gan = 0

        for i in range(self.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2) # each layer is mutliplied by two

        for i in range(self.num_layers):
            seq = []
            # generate list of neighbours at each topology (varies bc of Pooling)
            neighbor_list = skeleton.find_neighbor(self.topologies[i], self.skeleton_dist) 
            in_channels = self.channel_base[i] * self.joint_num[i]
            out_channels = self.channel_base[i+1] * self.joint_num[i]

            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if i < self.num_layers - 1: bias = False
            else: bias = True # if last layer 

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.joint_num[i], bias=bias))

            # might not need that
            if i < self.num_layers - 1: seq.append(torch.nn.BatchNorm1d(out_channels))

            pool = SkeletonPool(edges=self.topologies[i], pooling_mode='mean',
                                channels_per_edge=out_channels // len(neighbor_list))
            seq.append(pool)

            if not self.patch_gan or i < self.num_layers - 1: # not last layer 
                seq.append(torch.nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(torch.nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.joint_num.append(len(pool.new_edges) + 1)

            
            if i == self.num_layers - 1:
                self.last_channel = self.joint_num[-1] * self.channel_base[i+1]
        
        # if we get to the last layer, compress every two frames together
        # First fully connected layer
        self.fc1 = torch.nn.Linear(self.last_channel*2, 16)
      # Second fully connected layer that outputs our 10 labels
        self.fc2 = torch.nn.Linear(16, 1)
        # if not self.patch_gan: self.compress = torch.nn.Linear(in_features=self.last_channel, out_features=1)

    def forward(self, input):
        
        # print("[DISCRIM] input shape : ", input.shape)
        
        input = input.reshape(input.shape[0], -1)
        # input = input.permute(0, 2, 1)
        # print("[DISCRIM] input reshape : ", input.shape)

        for i,layer in enumerate(self.layers):
            # print("[DISCRIM] layer input shape : ", input.shape)

            input = layer(input)
            # print("[DISCRIM] layer {} output : {}".format(i,input.shape))
        
        # if not self.patch_gan:
        #     input = input.reshape(input.shape[0], -1)
        #     input = self.compress(input)
        # shape = (64, 72, 9)

        
        input = input.reshape(input.shape[0]//2,-1)
        # print("[DISCRIM] layer input reshape : ", input.shape)
        input = self.fc1(input)
        input = torch.nn.functional.relu(input)
        input = self.fc2(input)
        
        return torch.sigmoid(input).squeeze()