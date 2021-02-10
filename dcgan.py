#Importing Librarires

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils


############################
# IMprtant PH
CUDA = True
#DATA_PATH = '~/Data/mnist'
DATA_PATH = '.'
OUT_PATH = 'output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 25
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1
############################


# utils.clear_folder(OUT_PATH) will empty the output folder for us and 
# create one if it doesn't exist. 
utils.clear_folder(OUT_PATH)
print("Logging to {}\n".format(LOG_FILE))

# sys.stdout = utils.StdOut(LOG_FILE) will redirect all messages from print to the log file 
# and show these messages in the console at the same time.
sys.stdout = utils.StdOut(LOG_FILE)

CUDA = CUDA and torch.cuda.is_available()
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))
if seed is None:
    seed = np.random.randint(1, 10000)

print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)

# cudnn.benchmark = True will tell cuDNN to choose the best set of algorithms for your model 
# if the size of input data is fixed; 
# otherwise, cuDNN will have to find the best algorithms at each iteration.    
cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")








############# ###    GENERATOR NETWORK    ### ###############

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            #Applies a 2D transposed convolution operator over an input image composed of several input planes.
            #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel,, stride, padding)
            #bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            
            # 2nd layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            
            # 3rd layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            
            # 4th layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
#-------------------------------------------------------

# create a helper function to initialize the network parameters:
# These weight s will be used to initialize convolution kernels based on guassian distr.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#---------------------------

# Creating a generator object
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
#########################################################################################






############# ###    Discriminator NETWORK    ### ###############

class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            # Layer one
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, True),

            #Layer two
            nn.Conv2d(D_HIDDEN, D_HIDDEN *2, 4,2,1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, True),

            #Layer three
            nn.Conv2d(D_HIDDEN*2, D_HIDDEN*4, 4, 2, 1, bias= False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, True),

            #Layer four
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4,2,1, bias=False),
            nn.BatchNorm2d(D_HIDDEN*8),
            nn.LeakyReLU(0.2, True),

            #Output Laayer
            nn.Conv2d(D_HIDDEN*8, 1, 4,1,0, bias = False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)
#########################################################################



#Loss
criterion = nn.BCELoss()
#Optimizers
optimG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5,0.999))
optimD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))



#Loading Dataset
dataset = dset.MNIST (
    root=DATA_PATH, download=True, 
transform=transforms.Compose([transforms.Resize(X_DIM), transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,) )])
)

assert dataset
dataloader = torch.utils.data.DataLoader (dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# not sure why it is here
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

################   TRAINGING START HERE   ####################

for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader):
        x_real = data[0].to(device)

        print ("data[0] is:  ", data[0])
        
        # Discrimintator must learn for real and fake images being fed to it.
        real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device, dtype= torch.float) 
        fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device, dtype=torch.float)

        # Update *** Discriminator *** with real data
       
        # zerioing gradients before backpropogation
        netD.zero_grad()
        # feeding data to dicriminator
        y_real = netD(x_real)
        #calculating loss
        loss_D_real = criterion(y_real, real_label)
        #doing backward
        loss_D_real.backward()

        # Update *** Dicriminator *** with Fake Data
        
        # change with (if not work :         z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device))
        z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
        x_fake = netG(z_noise)
        # We are detaching because we want only DISC. to update. We DONT want generator to update its weights.
        y_fake = netD(x_fake.detach())
        loss_D_fake = criterion(y_fake, fake_label)
        loss_D_fake.backward()
        # call optimizer to update Discriminator parameters.
        optimD.step()




        # Update G with fake data
        netG.zero_grad()
        y_fake_r = netD(x_fake)
        loss_G = criterion(y_fake_r, real_label)
        loss_G.backward()
        optimG.step()




        if i % 100 == 0:
            print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                epoch, i, len(dataloader),
                loss_D_real.mean().item(),
                loss_D_fake.mean().item(),
                loss_G.mean().item()
            ))
            vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
            with torch.no_grad():
                viz_sample = netG(viz_noise)
                vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
    torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
    torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))