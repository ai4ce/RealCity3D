"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
# import wandb

# from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import sample
from thop import profile
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.log_dir = args.log_dir
        self.model_path = args.model_path
        # print("Using wandb")
        # wandb.init(project='LayoutTransformer', name=args.exp)
        # wandb.config.update(args)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self,epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint_{epoch}.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        pretrain_model = torch.load(self.model_path)
        raw_model.load_state_dict(pretrain_model)
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1
        data = self.test_dataset #if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        raw_model = raw_model.to(self.device)
        length = len(loader)
        for it, (x, y) in enumerate(loader):
            print('total: ',length, 'now: ',it)

            x = x.to(self.device)
            logits, _ = raw_model(x)
            # test_input = torch.ones(x.shape)
            flops, params = profile(raw_model, inputs=(x,))
            print('flops: ',flops)
            print('params: ',params)
            
            #print('logits.shape: ',logits.shape)
            probs = F.softmax(logits, dim=-1)
            _, out = torch.topk(probs, k=1, dim=-1)
            save_path = self.config.samples_dir+f'/{it}_model/predicted'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #print('out1.shape: ',out.shape)
            out = out.reshape(out.size(0),-1)
            #print('out.shape: ',out.shape)
            out = out.detach().cpu().numpy()
            for i, layout in enumerate(out):
                #print('layout.shape: ',layout.shape)
                self.train_dataset.render(layout,save_path,i)
            #print('y.shape: ',y.shape)
            # y = y.reshape(32,-1)
            save_path_gt = self.config.samples_dir+f'/{it}_model/gt'
            if not os.path.exists(save_path_gt):
                os.makedirs(save_path_gt)
            y = y.detach().cpu().numpy()
            for i, layout in enumerate(y):
                #print('layout.shape: ',layout.shape)
                self.train_dataset.render(layout,save_path_gt,i)
        '''flops:  1045.485.715.456.0
            params:  25385472.0'''
        # # def run_epoch(split,writer):
        # #     is_train = split == 'train'
        # #     model.train(is_train)
        # #     data = self.train_dataset if is_train else self.test_dataset
        # #     loader = DataLoader(data, shuffle=True, pin_memory=True,
        # #                         batch_size=config.batch_size,
        # #                         num_workers=config.num_workers)

        # #     losses = []
        # #     #pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        # #     for it, (x, y) in enumerate(loader):

        # #         if epoch == 0 and not is_train:
        # #             self.fixed_x = x[:min(5, len(x))]
        # #             self.fixed_y = y[:min(5, len(y))]

        # #         # place data on the correct device
        # #         x = x.to(self.device)
        # #         y = y.to(self.device)

        # #         # forward the model
        # #         with torch.set_grad_enabled(is_train):
        # #             # import ipdb; ipdb.set_trace()
        # #             logits, loss = model(x, y, pad_token=pad_token)
        # #             loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
        # #             #print('loss.item(): ',loss.item())
        # #             losses.append(loss.item())

        # #         if is_train:

        # #             # backprop and update the parameters
        # #             model.zero_grad()
        # #             loss.backward()
        # #             torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        # #             optimizer.step()
        # #             self.iters += 1
        # #             # decay the learning rate based on our progress
        # #             if config.lr_decay:
        # #                 # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
        # #                 if self.iters < config.warmup_iters:
        # #                     # linear warmup
        # #                     lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
        # #                 else:
        # #                     # cosine learning rate decay
        # #                     progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
        # #                     lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # #                 lr = config.learning_rate * lr_mult
        # #                 for param_group in optimizer.param_groups:
        # #                     param_group['lr'] = lr
        # #             else:
        # #                 lr = config.learning_rate
        # #             writer.add_scalar('loss_train', loss.item(), global_step=epoch)
        # #             writer.add_scalar('lr', lr, global_step=epoch)
        # #             #print('loss_train: ',loss.item())
        # #             # report progress
        # #             # wandb.log({
        # #             #     'train loss': loss.item(),
        # #             #     'lr': lr, 'epoch': epoch+1
        # #             # }, step=self.iters)
        # #             #pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        # #     if not is_train:
        # #         test_loss = float(np.mean(losses))
        # #         logger.info("test loss: %f", test_loss)
        # #         writer.add_scalar('test_loss', test_loss, global_step=epoch)
        # #         # wandb.log({'test loss': test_loss}, step=self.iters)
        # #         return test_loss
        # # writer = SummaryWriter(self.log_dir)

        # # best_loss = float('inf')
        # for epoch in range(config.max_epochs):
        #     run_epoch('train',writer)
        #     if self.test_dataset is not None:
        #         with torch.no_grad():
        #             test_loss = run_epoch('test',writer)

        #     # supports early stopping based on the test loss, or just save always if no test set is provided
        #     good_model = self.test_dataset is None or test_loss < best_loss
        #     if self.config.ckpt_dir is not None and good_model:
        #         best_loss = test_loss
        #         self.save_checkpoint(epoch)
        #     elif epoch%100==0:
        #         self.save_checkpoint(epoch)
            

        #     # sample from the model
        #     if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
        #         # inputs
        #         layouts = self.fixed_x.detach().cpu().numpy()
        #         for i, layout in enumerate(layouts):
        #             #print('layout: ',layout.shape)
        #             save_path = self.config.samples_dir+f'/{epoch}'
        #             if not os.path.exists(save_path):
        #                 os.makedirs(save_path)
        #             self.train_dataset.render(layout,save_path,i)
                    
                    
        #         # reconstruction
        #         x_cond = self.fixed_x.to(self.device)
        #         logits, _ = model(x_cond)
        #         #print('logits.shape: ',logits.shape)
        #         probs = F.softmax(logits, dim=-1)
        #         _, y = torch.topk(probs, k=1, dim=-1)
        #         #print('y.shape: ',y.shape)
        #         #print('x_cond[:, :1]: ',x_cond[:, :1].shape)
        #         #print('y[:, :, 0].shape: ',y[:, :, 0].shape)
        #         # layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
        #         #recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
        #         save_path = self.config.samples_dir+f'/{epoch}_model'
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #         y = y.reshape(y.size(0),-1)
        #         #print('y.shape: ',y.shape)
        #         y = y.detach().cpu().numpy()
        #         for i, layout in enumerate(y):
        #             #print('layout.shape: ',layout.shape)
        #             self.train_dataset.render(layout,save_path,i)
        #             # layout.save(os.path.join(self.config.samples_dir+f'/{epoch}', f'recon_{epoch:02d}_{i:02d}.png'))
                
        #         # # samples - random
        #         # layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
        #         #                  temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        #         # sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
        #         # for i, layout in enumerate(layouts):
        #         #     if not os.path.exists(self.config.samples_dir+f'/{epoch}'):
        #         #         os.makedirs(self.config.samples_dir+f'/{epoch}')
        #         #     layout = self.train_dataset.render(layout)
        #         #     layout.save(os.path.join(self.config.samples_dir+f'/{epoch}', f'sample_random_{epoch:02d}_{i:02d}.png'))

        #         # # samples - deterministic
        #         # layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
        #         #                  temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
        #         # sample_det_layouts = [self.train_dataset.render(layout) for layout in layouts]
        #         # for i, layout in enumerate(layouts):
        #         #     if not os.path.exists(self.config.samples_dir+f'/{epoch}'):
        #         #         os.makedirs(self.config.samples_dir+f'/{epoch}')
        #         #     layout = self.train_dataset.render(layout)
        #         #     layout.save(os.path.join(self.config.samples_dir+f'/{epoch}', f'sample_det_{epoch:02d}_{i:02d}.png'))

        #         # wandb.log({
        #         #     "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
        #         #                       for i, pil in enumerate(input_layouts)],
        #         #     "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
        #         #                       for i, pil in enumerate(recon_layouts)],
        #         #     "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
        #         #                               for i, pil in enumerate(sample_random_layouts)],
        #         #     "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
        #         #                            for i, pil in enumerate(sample_det_layouts)],
        #         # }, step=self.iters)
