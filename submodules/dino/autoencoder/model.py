import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, model_name):
        super(Autoencoder, self).__init__()
        if 'dino_affine' in model_name:
            init_dim = 384
            decoder_hidden_dims = [init_dim]
        elif 'dino' in model_name:
            init_dim = 384
        elif 'clip' in model_name:
            init_dim = 512
        elif 'devit' in model_name:
            init_dim = 768
        elif 'sd' in model_name:
            init_dim = 2560
            encoder_hidden_dims = [2560, 1280, 640, 320, 16]
            decoder_hidden_dims = [320, 1280, 2560, 2560, 2560]
        


        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Conv2d(init_dim, encoder_hidden_dims[i],1))
            else:
                # encoder_layers.append(torch.nn.BatchNorm2d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Conv2d(encoder_hidden_dims[i-1], encoder_hidden_dims[i],1))
        self.encoder = nn.ModuleList(encoder_layers)
        self.bn = torch.nn.BatchNorm2d(encoder_hidden_dims[-1])

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Conv2d(encoder_hidden_dims[-1], decoder_hidden_dims[i], 1))
            else:
                # encoder_layers.append(torch.nn.BatchNorm2d(decoder_hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Conv2d(decoder_hidden_dims[i-1], decoder_hidden_dims[i],1))
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder, self.decoder)
    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        # x = x / x.norm(dim=-1, keepdim=True)
        self.bn(x)
        return x
    
    def encode_normalize(self, x):
        for m in self.encoder:
            x = m(x)    
        # x = x / x.norm(dim=-1, keepdim=True)
        
        return self.bn(x)

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        # x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def decode_denormalize(self, x):
        x = x * torch.sqrt(self.bn.running_var.view(x.shape[0],1,1)) + self.bn.running_mean.view(x.shape[0],1,1)
        for m in self.decoder:
            x = m(x)    

        return x
    
    