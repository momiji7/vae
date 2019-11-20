import torch
import math
import torch.nn.functional as F

'''
gaussian
def vae_loss(x, x_mu, x_sig, z_mu, z_sig):
    # max BLEO ---> min -BLEO
    nbatch , Dz = z_mu.size(0), z_mu.size(2)
    L, Dx = x_mu.size(1), x_mu.size(2)
    #loss = (-torch.sum(z_mu * z_mu) - torch.sum(z_sig) + torch.sum(torch.log(z_sig)) + nbatch*Dz)/2
    #loss += -torch.log(math.pi*2)*Dx*nbatch/2 - torch.sum((x - x_mu)*(x - x_mu)/x_sig)*/2/L
    #        -torch.sum(torch.log(x_sig))*/2/L
    #loss = -loss

    # remove the constant, if use the total loss, the encoder's parameters will be updated by the outputs of decoder
    loss = (-torch.sum(z_mu * z_mu) - torch.sum(z_sig) + torch.sum(torch.log(z_sig)))/2
    loss += -(torch.sum((x - x_mu)*(x - x_mu)/x_sig) + torch.sum(torch.log(x_sig)))/2/L
    loss = -loss
    
    decoder_loss = (torch.sum((x - x_mu)*(x - x_mu)/x_sig) + torch.sum(torch.log(x_sig)))/2/L
    #encoder_loss = (torch.sum(z_mu * z_mu) + torch.sum(z_sig) - torch.sum(torch.log(z_sig)))/2
    #decoder_loss = (torch.sum((x - x_mu)*(x - x_mu)/x_sig) + torch.sum(torch.log(x_sig)))/2/L
    
    #return encoder_loss, decoder_loss
    
    return loss, decoder_loss
'''


# bernoulli
def vae_loss(x, x_ber, z_mu, z_sig):
    # max BLEO ---> min -BLEO
    nbatch , Dz = z_mu.size(0), z_mu.size(2)
    L, Dx = x_ber.size(1), x_ber.size(2)
    
    loss = (-torch.sum(z_mu * z_mu) + torch.sum(z_sig) - torch.sum(torch.exp(z_sig)))/2
    loss += -F.binary_cross_entropy(x_ber, x, reduction='sum')
    loss = -loss

    
    decoder_loss = -0.5*torch.sum(1 + z_sig - z_mu.pow(2) - torch.exp(z_sig))

    #print(bcd)
    #print(decoder_loss)
    
    return loss, decoder_loss

