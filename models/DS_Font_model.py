import torch
from .base_model import BaseModel
from . import networks
import ttools.modules
from . import MSP
import torch.nn as nn
from . import cm
import itertools

class DSFontModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm='batch', netG='FTGAN_MLAN', dataset_mode='font')
        
        if is_train:
            parser.set_defaults(batch_size=256, pool_size=0, gan_mode='hinge', netD='basic_64')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
            parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for content loss')
            parser.add_argument('--lambda_G_NCE', type=float, default=0.5, help='weight for content loss')
            parser.add_argument('--nce_layers', type=str, default='0,1,2', help='compute NCE loss on which layers')
            parser.add_argument('--dis_2', default=False, help='use two discriminators or not')
            parser.add_argument('--use_spectral_norm', default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
            
        if self.isTrain:
            self.dis_2 = opt.dis_2
            self.visual_names = ['gt_images', 'generated_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]
            if self.dis_2:
                self.model_names = ['G', 'D_content', 'D_style','P','P_style']
                self.loss_names = ['G_GAN', 'G_L1', 'D_content', 'D_style']
            else:
                self.model_names = ['G', 'D','P','P_style']
                self.loss_names = ['G_GAN', 'G_L1', 'D','G_NCE','nce']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.style_channel+1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.dis_2:
                self.netD_content = networks.define_D(2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
                self.netD_style = networks.define_D(self.style_channel+1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            else:
                discriminator = networks.disc_builder(32,818,993)
                self.netD = networks.init_net(discriminator, opt.init_type, opt.init_gain, opt.gpu_ids)
            style_vgg = MSP.vgg
            #style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
            style_vgg = nn.Sequential(*list(style_vgg.children()))
            self.netP = MSP.StyleExtractor(style_vgg,self.gpu_ids )
            self.netP_style = MSP.Projector(self.gpu_ids)  
            self.netP = networks.init_net(self.netP, 'normal', 0.02, self.gpu_ids) 
            self.netP_style = networks.init_net(self.netP_style, 'normal', 0.02, self.gpu_ids)
            self.memory = cm.ClusterMemory(2080,818,momentum=0.1)
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            self.lambda_NCE = opt.lambda_NCE
            self.optimizer_MSP = torch.optim.Adam(itertools.chain(self.netP.parameters(), self.netP_style.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_MSP)

        if self.isTrain:
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.dis_2:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_style = torch.optim.Adam(self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        self.cids = data['char_label']
        self.fids = data['font_label'] 
        if not self.isTrain:
            self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG((self.content_images, self.style_images))
        
    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake 
        fake, _ = netD(fake_images.detach(),self.fids,self.cids)
        loss_D_fake = self.criterionGAN(fake[0], False)+self.criterionGAN(fake[1], False)
        # Real
        real,_ = netD(real_images,self.fids,self.cids)
        loss_D_real = self.criterionGAN(real[0], True)+ self.criterionGAN(real[1], True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images,real_images, netD):
        fake,_ = netD(fake_images,self.fids,self.cids)
        #real,_ = netD(real_images.detach(),self.fids,self.cids)
        loss_G_GAN = self.criterionGAN(fake[0],True)+self.criterionGAN(fake[1],True)
        return loss_G_GAN
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.dis_2:
            self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images],  [self.content_images, self.generated_images], self.netD_content)
            self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [self.style_images, self.generated_images], self.netD_style)
            self.loss_D = self.lambda_content*self.loss_D_content + self.lambda_style*self.loss_D_style         
        else:
            self.loss_D = self.compute_gan_loss_D(self.gt_images,  self.generated_images, self.netD)

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
    
        if self.dis_2:
            self.loss_G_content = self.compute_gan_loss_G([self.content_images, self.generated_images], self.netD_content)
            self.loss_G_style = self.compute_gan_loss_G([self.style_images, self.generated_images], self.netD_style)
            self.loss_G_GAN = self.lambda_content*self.loss_G_content + self.lambda_style*self.loss_G_style
        else:
            self.loss_G_GAN = self.compute_gan_loss_G(self.generated_images,self.gt_images, self.netD)

        if self.opt.lambda_G_NCE > 0:
            query_B = self.netP_style(self.netP(self.generated_images, self.nce_layers),self.nce_layers)   
            num = 0
            self.loss_G_NCE = 0
            for x in self.nce_layers:
                self.loss_G_NCE += self.memory(query_B[num],self.fids,x)
                num += 1
            self.loss_G_NCE = self.loss_G_NCE*self.opt.lambda_G_NCE
        else:
            self.loss_G_NCE = 0

        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1+self.loss_G_NCE
        self.loss_G.backward()
        
    def backward_NCEloss(self):
        real_gt = self.netP((self.gt_images),self.nce_layers) 
        query = self.netP_style(real_gt,self.nce_layers)
        num = 0
        self.loss_nce = 0
        for x in self.nce_layers:
            self.loss_nce += self.memory(query[num], self.fids,x,grad=1)
            #self.nce_loss.dequeue_and_enqueue(key[num], K,'real_B{:d}'.format(x))
            num += 1
        
        return self.loss_nce*self.opt.lambda_NCE
    
    def optimize_parameters(self):
        torch.cuda.empty_cache()
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.set_requires_grad([self.netP, self.netP_style,self.netG], False)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()
        else:
            self.set_requires_grad(self.netD, True)      # enable backprop for D
            self.set_requires_grad([self.netP, self.netP_style,self.netG], False)
            self.optimizer_D.zero_grad()             # set D's gradients to zero
            self.backward_D()                    # calculate gradients for D
            self.optimizer_D.step()                # update D's weights
        # update MSP
        if self.opt.lambda_NCE>0:
            if self.dis_2:
                self.set_requires_grad([self.netG,self.netD_content, self.netD_style], False)
                self.set_requires_grad([self.netP,self.netP_style], True)
            else:
                self.set_requires_grad([self.netG,self.netD], False)
                self.set_requires_grad([self.netP,self.netP_style], True)
            self.optimizer_MSP.zero_grad()
            self.loss_NCE = self.backward_NCEloss()
            self.loss_NCE.backward(retain_graph=True)
            self.optimizer_MSP.step()
        # update G
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style,self.netP,self.netP_style], False)
            self.set_requires_grad([self.netG], True)
        else:
            self.set_requires_grad([self.netD,self.netP,self.netP_style], False)  # D requires no gradients when optimizing G
            self.set_requires_grad([self.netG], True)
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(self, 'style_images_{}'.format(i), torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train()
        else:
            pass    
