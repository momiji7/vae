import torch
from network import *
from datasets import *
from loss import *
import numpy as np
import cv2
from tensorboardX import SummaryWriter
from meter import AverageMeter
from logger import Logger
import argparse
import datetime
import scipy.misc

tfboard_writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument("--img_size",     default=28, type=int,   help="size of image")
parser.add_argument("--latent_dim",   default=10, type=int,    help="dimension of latent space")
parser.add_argument("--batch_size",   default=8,  type=int,  help="minibatch size")
parser.add_argument("--epoch",        default=20, type=int,    help="training epoch")
parser.add_argument("--encoder_lr",   default=1e-7,type=float,  help="learning rate")
parser.add_argument("--decoder_lr",   default=1e-7,type=float,  help="learning rate")
parser.add_argument("--sample_times", default=1  ,type=int,    help="MC sample times")
parser.add_argument("--test_latent_num", default=10  ,type=int,    help="the number of latent space in test stage")
parser.add_argument("--test_sample_times", default=10  ,type=int,  help="sample times in test stage")
parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
parser.add_argument('--save_path',        type=str,   default='./snapshots/', help='Folder to save checkpoints and log.')

args=parser.parse_args()

logname = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
logger = Logger(args.save_path, logname)
logger.log('Arguments : -------------------------------')
for name, value in args._get_kwargs():
    logger.log('{:16} : {:}'.format(name, value))

vae_encoder = encoder(args)
vae_decoder = decoder(args)

train_dataset = vae_dataset(args)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

opt_encoder = torch.optim.SGD(vae_encoder.parameters(), lr=args.encoder_lr, momentum=0.9, weight_decay=0.0005)
sch_encoder = torch.optim.lr_scheduler.StepLR(opt_encoder, step_size=20, gamma=0.1)

opt_decoder = torch.optim.SGD(vae_decoder.parameters(), lr=args.decoder_lr, momentum=0.9, weight_decay=0.0005)
sch_decoder = torch.optim.lr_scheduler.StepLR(opt_decoder, step_size=20, gamma=0.1)

vae_decoder = vae_decoder.cuda()
vae_encoder = vae_encoder.cuda()

logger.log("=> encoder network :\n {}".format(vae_encoder))
logger.log("=> decoder network :\n {}".format(vae_decoder))

last_info = logger.last_info()
if last_info.exists():
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
    vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
    opt_encoder.load_state_dict(checkpoint['opt_encoder'])
    opt_decoder.load_state_dict(checkpoint['opt_decoder'])
    sch_encoder.load_state_dict(checkpoint['sch_encoder'])
    sch_decoder.load_state_dict(checkpoint['sch_decoder'])
    logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0

for ep in range(start_epoch, args.epoch):
    sch_encoder.step()
    sch_decoder.step()

    vae_encoder.train()
    vae_decoder.train()
    
    encoder_losses = AverageMeter()
    decoder_losses = AverageMeter()
    # train
    for ibatch, (img ,noise) in enumerate(train_loader):
        
        img = img.cuda()
        noise = noise.cuda()
        
        z_mu, z_sig = vae_encoder(img) # z_mu , z_sig : N*1*Dz
        zl = torch.sqrt(z_sig) * noise + z_mu      # zl           : N*L*Dz
        x_mu, x_sig = vae_decoder(zl)  # x_mu , x_sig : N*L*Dx
      
        encoder_loss, decoder_loss = vae_loss(img, x_mu, x_sig, z_mu, z_sig)
        
        encoder_losses.update(encoder_loss.item())
        decoder_losses.update(decoder_loss.item())

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        encoder_loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        
        
        if ibatch % args.print_freq == 0 or ibatch+1 == len(train_loader):
            logger.log('[train Info]: [epoch-{}-{}][{:04d}/{:04d}][Encoder Loss:{:.2f}][Decoder Loss:{:.2f}]'.format(ep, args.epoch, ibatch, len(train_loader), encoder_loss.item(), decoder_loss.item()))


    tfboard_writer.add_scalar('Encoder_Loss', encoder_losses.avg, ep)
    tfboard_writer.add_scalar('Decoder_Loss', decoder_losses.avg, ep)
    logger.log('epoch {:02d} completed!'.format(ep))
    logger.log('[train Info]: [epoch-{}-{}][Avg Encoder Loss:{:.6f}][Avg Decoder Loss:{:.6f}]'.format(ep, args.epoch, encoder_losses.avg, decoder_losses.avg))

    filename = 'epoch-{}-{}.pth'.format(ep, args.epoch)
    save_path = logger.path('model') / filename
    torch.save({
      'epoch': ep,
      'args' : args,
      'encoder_state_dict': vae_encoder.state_dict(),
      'decoder_state_dict': vae_decoder.state_dict(),
      'sch_encoder' : sch_encoder.state_dict(),
      'sch_decoder' : sch_decoder.state_dict(),
      'opt_encoder' : opt_encoder.state_dict(),
      'opt_decoder' : opt_decoder.state_dict(),
    }, logger.path('model') / filename)
    logger.log('save checkpoint into {}'.format(filename))
    last_info = torch.save({
      'epoch': ep,
      'last_checkpoint': save_path
    }, logger.last_info())


    # test
    img = np.zeros((args.test_latent_num*args.img_size, args.test_sample_times*args.img_size), dtype=np.float32)
    with torch.no_grad():
        vae_encoder.eval()
        vae_decoder.eval()
        for itest in range(args.test_latent_num):
            z = torch.randn(1, args.latent_dim).cuda()
            x_mu, x_sig = vae_decoder(z) # 1*Dx

            
            n = torch.randn(args.test_sample_times, args.img_size*args.img_size).cuda()
            x_systhesis = n * torch.sqrt(x_sig) + x_mu # ST*Dx
            x_systhesis = x_systhesis.reshape(args.test_sample_times, args.img_size, args.img_size)
            x_systhesis = x_systhesis.to(torch.device('cpu'))
            x_systhesis = x_systhesis.numpy()
            for isample in range(args.test_sample_times):
                img[itest*args.img_size:(itest+1)*args.img_size, isample*args.img_size:(isample+1)*args.img_size] = x_systhesis[isample] * 0.3081 + 0.1307

    scipy.misc.toimage(img,cmin=0.0,cmax=1.0).save('{}/epoch_{:03d}_sample.jpg'.format(logger.path('model'), ep))
   
logger.close()








