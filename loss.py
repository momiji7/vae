import torch
import math

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

