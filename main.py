import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiModalVAE(nn.Module):
   def __init__(self, input_dims, h_dim, z_dim):
       super(MultiModalVAE, self).__init__()
       self.input_dims = input_dims
       self.z_dim = z_dim


       self.encoders_conv = nn.ModuleList([
           nn.Sequential(
               nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
               nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
               nn.ReLU(),
               nn.Flatten()
           ) for _ in input_dims
       ])


       conv_output_dim = 64 * 14 * 14  # 64 channels with 14x14 feature map after two conv layers
       self.encoders_fc = nn.ModuleList([
           nn.Linear(conv_output_dim, h_dim) for _ in input_dims
       ])
       self.fc_mu = nn.ModuleList([
           nn.Linear(h_dim, z_dim) for _ in input_dims
       ])
       self.fc_logvar = nn.ModuleList([
           nn.Linear(h_dim, z_dim) for _ in input_dims
       ])


       self.decoders_fc = nn.ModuleList([
           nn.Linear(z_dim, h_dim) for _ in input_dims
       ])
       self.decoders_conv = nn.ModuleList([
           nn.Sequential(
               nn.Linear(h_dim, conv_output_dim),
               nn.ReLU(),
               nn.Unflatten(1, (64, 14, 14)),
               nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
               nn.ReLU(),
               nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
               nn.Sigmoid()
           ) for _ in input_dims
       ])


   def encode(self, x):
       mu_list = []
       log_var_list = []
       for i in range(len(self.input_dims)):
           if x[i] is not None:
               h = self.encoders_conv[i](x[i].view(-1, 1, 28, 28))
               h = self.encoders_fc[i](h)
               mu_list.append(self.fc_mu[i](h))
               log_var_list.append(self.fc_logvar[i](h))
           else:
               mu_list.append(None)
               log_var_list.append(None)
       return mu_list, log_var_list


   def reparameterize(self, mu, log_var):
       std = torch.exp(0.5 * log_var)
       eps = torch.randn_like(std)
       return mu + eps * std


   def decode(self, z):
       recons_list = []
       for i in range(len(self.input_dims)):
           h = self.decoders_fc[i](z)
           recons_list.append(self.decoders_conv[i](h))
       return recons_list


   def forward(self, x):
       mu_list, log_var_list = self.encode(x)
       z_list = [self.reparameterize(mu, log_var) for mu, log_var in zip(mu_list, log_var_list)]
       recons_list = self.decode(z_list[0])  # Using the first latent vector for simplicity
       return recons_list, mu_list, log_var_list, z_list

