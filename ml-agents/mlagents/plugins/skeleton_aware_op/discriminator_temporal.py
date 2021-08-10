import skeleton_temporal
from skeleton_temporal import SkeletonConv, SkeletonPool
from mlagents.torch_utils import torch, default_device
from options import get_options


class Discriminator(torch.nn.Module):
    def __init__(self, topology, real_label = 1, fake_label = 0, criterion_gan = None):

        super(Discriminator, self).__init__()

        options = get_options()

        self.topologies = [topology]
        self.channel_base = [3] # work on position?
        self.channel_list = []
        self.joint_num = [len(topology) + 1] # edges + 1
        self.pooling_list = []
        self.layers = torch.nn.ModuleList()
        self.num_layers = options['num_layers']
        self.skeleton_dist = options['num_neighbours']
        self.patch_gan = 0
        kernel_size = options['kernel_size']
        padding = (kernel_size - 1) // 2

        self.real_label = real_label
        self.fake_label = fake_label

        if criterion_gan is None:
            self.criterion_gan = torch.nn.MSELoss()

        for i in range(self.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2) # each layer is mutliplied by two

        for i in range(self.num_layers):
            seq = []
            # generate list of neighbours at each topology (varies bc of Pooling)
            neighbor_list = skeleton_temporal.find_neighbor(self.topologies[i], self.skeleton_dist) 
            in_channels = self.channel_base[i] * self.joint_num[i]
            out_channels = self.channel_base[i+1] * self.joint_num[i]

            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if i < self.num_layers - 1: bias = False
            else: bias = True # if last layer 

            if i == self.num_layers - 1:
                kernel_size = options['kernel_size']
                padding = 0

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.joint_num[i], kernel_size=options['kernel_size'], stride=2, padding=padding,
                                    padding_mode='reflection', bias=bias))

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
                # self.last_channel = self.joint_num[-1] * self.channel_base[i+1]
                self.last_channel = 168
        
        # if we get to the last layer, compress every two frames together
        # First fully connected layer
        # self.fc1 = torch.nn.Linear(self.last_channel*2, 16)
      # Second fully connected layer that outputs our 10 labels
        # self.fc2 = torch.nn.Linear(16, 1)
        if not self.patch_gan: self.compress = torch.nn.Linear(in_features=self.last_channel, out_features=1)

    def forward(self, input):

        input = input.reshape(input.shape[0], input.shape[1], -1)
        input = input.permute(0, 2, 1)
        for layer in self.layers:
            input = layer(input)
        if not self.patch_gan:
            input = input.reshape(input.shape[0], -1)

            input = self.compress(input)
        return torch.sigmoid(input).squeeze()


    def D_loss(self, real_pos, fake_pos, isTest):

        curr_batch_size = real_pos.shape[0]
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=default_device())
        
        if not isTest: self.zero_grad()

        # forward pass
        output = self.forward(real_pos.float()).view(-1)
        # calculate the loss and gradients
        loss_real = self.criterion_gan(output,label)
        if not isTest : loss_real.backward()
        
        # do the same with the generated position
        label.fill_(self.fake_label)
        # forward pass
        output = self.forward(fake_pos.float().detach()).view(-1)
        # calculate loss and gradients
        loss_fake = self.criterion_gan(output, label)
        if not isTest : loss_fake.backward()

        loss_discrim = loss_real + loss_fake

        return loss_real, loss_fake, loss_discrim
        
    def G_loss(self, fake_pos):
        """
        Compute the least square loss for the generator.
        Try to maximize log(D(G(x)))
        """
        curr_batch_size = fake_pos.shape[0]

        # Adversial Loss, use real labels to maximize log(D(G(x))) instead of log(1-D(G(x)))
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=device)
        label.fill_(self.real_label)
        output = self.forward(fake_pos.float()).view(-1)
        loss_adv = self.criterion_gan(output,label)

        return loss_adv
