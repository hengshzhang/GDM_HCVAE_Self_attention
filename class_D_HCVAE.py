import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
# import torchkeras
Cardinal = 8
level = 4
Hide_layer_size = 28

class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size: object, class_size: object, latent_size: object) -> object:
        super(CVAE, self).__init__() # “”“这段代码的含义是首先找到CVAE的父类，然后将CVAE类的对象转化为父类的对象，
        # 让后让这个“被转化”的对象调用自己的__init__()函数”“”
        self.fc1 = nn.Linear(feature_size + class_size, 100)
        self.fc2_mu = nn.Linear(100, latent_size)
        self.fc2_log_std = nn.Linear(100, latent_size)
        self.fc3 = nn.Linear(latent_size + class_size, 100)
        self.fc4 = nn.Linear(100, feature_size)
        self.fc5 = nn.Linear(latent_size, 100)
        self.fc6_mu = nn.Linear(100, latent_size)
    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=0)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std_t = self.fc2_log_std(h1)
        log_std = torch.sigmoid(log_std_t)
        return mu, log_std
    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=0)))  # concat latents and labels
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon
    def reparametrize(self, mu, log_std):
        #std = torch.exp(log_std)
        std = log_std
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z
    def forward(self, x, y):
        mu_e = torch.zeros(level,Hide_layer_size)
        mu_d = torch.zeros(level,Hide_layer_size)
        logstd = torch.zeros(level,Hide_layer_size)
        #mu_e[0], logstd[0] = self.encode(x, y)
        mu0, log_std0 = self.encode(x, y)
        mu_e[0] = mu0
        logstd[0] = log_std0
        z1 = self.reparametrize(mu0, log_std0)
        discrete_z1 = self.latent_discrete(z1)
        mu1, log_std1 = self.encode(discrete_z1, y)
        mu_e[1] = mu1
        logstd[1] = log_std1
        z2 = self.reparametrize(mu1, log_std1)
        discrete_z2 = self.latent_discrete(z2)
        mu2, log_std2 = self.encode(discrete_z2, y)
        mu_e[2] = mu2
        logstd[2] = log_std2
        z3 = self.reparametrize(mu2, log_std2)
        discrete_z3 = self.latent_discrete(z3)
        mu3, log_std3 = self.encode(discrete_z3, y)
        z4 = self.reparametrize(mu3, log_std3)
        discrete_z4 = self.latent_discrete(z4)
        recon3 = self.decode(discrete_z4, y)
        discrete_recon3 = self.latent_discrete(recon3)
        mu_d3 = self.fc6_mu(self.fc5(recon3))
        mu_d[3] = mu_d3
        recon2 = self.decode(discrete_recon3, y)
        discrete_recon2 = self.latent_discrete(recon2)
        mu_d2 = self.fc6_mu(self.fc5(recon2))
        mu_d[2] = mu_d2
        recon1 = self.decode(discrete_recon2, y)
        discrete_recon1 = self.latent_discrete(recon1)
        mu_d1 = self.fc6_mu(self.fc5(recon1))
        mu_d[1] = mu_d1
        recon0 = self.decode(discrete_recon1, y)
        mu_d0 = self.fc6_mu(self.fc5(recon0))
        mu_d[0] = mu_d0
        recon = recon0
        # print("final_recon:", final_recon,"\n")
        # print("mu_e", mu_e,"\n")
        # print("mu_d", mu_d,"\n")
        # print("log_std",log_std,"\n")
        return recon, mu_e, mu_d, logstd


    def loss_function(self, recon, x, mu_e, mu_d, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = 0
        for i in range(level-1):
            temp_kl_loss = torch.pow((mu_e[i] - mu_d[level - 2 - i]), 2) - log_std[i]
            kl_loss = kl_loss + temp_kl_loss
        #kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss -= log_std[level-1]
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

    def trans_func(self,x):
        y = pow(math.sqrt(2),x-Cardinal/2)
        return y

    def calculat_nearest_one(self, x, vector):
        temp_x = x.detach().numpy()
        vector_len =len(vector)
        distance_iter = torch.zeros(vector_len)
        for i in range(vector_len):
            distance_iter[i] = torch.tensor(np.linalg.norm(temp_x - vector[i].numpy()))
        min_index = distance_iter.argmin()
        nearest_x = vector[min_index]
        return nearest_x


    def latent_discrete(self,z):
        discrete_vector = torch.tensor([0.2500,0.3536,0.5000,0.7071,1.0000,1.4142,\
                                    2.0000,2.8284,4.0000])
        latent_len = len(z)
        discrete_z = torch.zeros(latent_len)
        for j in range(latent_len):
            discrete_z[j] = self.calculat_nearest_one(z[j],discrete_vector)
        return discrete_z
#model = torchkeras.kerasmodel(CVAE(40,40,40))
#print(model)

