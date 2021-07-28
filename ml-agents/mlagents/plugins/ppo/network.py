from mlagents.torch_utils import torch, default_device


class MLPNet(torch.nn.Module):
    def __init__(self, input_dim:int,
                 output_dim:int,
                 hidden_dim:int=64,
                 num_layers:int=1, 
                 mid_activation = torch.nn.Tanh(),
                 last_activation=torch.nn.Tanh()):

        super(MLPNet, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        
        seq = []
        seq.append(torch.nn.Linear(input_dim, hidden_dim))
        seq.append(mid_activation)

        for i in range(num_layers):
            seq.append(torch.nn.Linear(hidden_dim, hidden_dim))
            seq.append(mid_activation)
        
        seq.append(torch.nn.Linear(hidden_dim, output_dim))
        seq.append(last_activation)

        self.model = torch.nn.Sequential(*seq)

        
    def forward(self, input):
        output = self.model.forward(input)
        return output


if __name__ == "__main__":
    actor = MLPNet(40,10,20,last_activation=torch.nn.Identity())
    res = actor.forward(torch.rand((40,),dtype=torch.float32))
    print(res)