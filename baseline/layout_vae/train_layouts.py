import os
import numpy as np
import argparse
import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datetime import datetime
from PIL import Image, ImageDraw
import seaborn as sns
from thop import profile

from box import AutoregressiveBoxEncoder, AutoregressiveBoxDecoder
from layout_my import BatchCollator, LayoutDataset
import logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


def plot_layout(layout,save_path,i):
    # blank_image = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    # blank_draw = ImageDraw.Draw(blank_image)

    # number_boxes = real_boxes.shape[0]
    # for i in range(number_boxes):
    #     real_box = real_boxes[i].tolist()
    #     predicted_box = predicted_boxes[i].tolist()
    #     label = int(labels[i])

    #     real_x1, real_y1 = int(real_box[0] * width), int(real_box[1] * height)
    #     real_x2, real_y2 = real_x1 + int(real_box[2] * width), real_y1 + int(real_box[3] * height)

    #     predicted_x1, predicted_y1 = int(predicted_box[0] * width), int(predicted_box[1] * height)
    #     predicted_x2, predicted_y2 = predicted_x1 + int(predicted_box[2] * width), predicted_y1 + int(
    #         predicted_box[3] * height)

    #     real_color = (0, 0, 0)
    #     if colors is not None:
    #         real_color = tuple(colors[label])

    #     blank_draw.rectangle([(real_x1, real_y1), (real_x2, real_y2)], outline=real_color)
    #     blank_draw.rectangle([(predicted_x1, predicted_y1), (predicted_x2, predicted_y2)], outline=(0, 0, 0)
    layout = layout.reshape(-1)
    # layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
    # logging.info(f'{save_path}: {layout}')
    # logging.info(f'{save_path}_shaep: {layout.shape}')
    box_r = layout[: len(layout) // 8 * 8].reshape(-1, 4, 2)
    # logging.info(f'box_r: {box_r}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #print('box_r.shape: ',box_r.shape)
    '''box_r.shape:  (32, 4, 2)'''
    plt.figure(figsize=(500,500),dpi=1)
    plt.axis('off')
    for box in box_r:
        plt.plot(np.concatenate((box[:,0],box[0:1,0]),axis=0), np.concatenate((box[:,1],box[0:1,1]),axis=0), "b",linewidth=100)
    savepath_image = save_path + '/image'
    if not os.path.exists(savepath_image):
        os.makedirs(savepath_image)
    plt.savefig(savepath_image+f'/{i}.png', dpi=1)
    savepath_npy = save_path +'/npy'
    if not os.path.exists(savepath_npy):
        os.makedirs(savepath_npy)
    np.save(savepath_npy+f'/{i}.npy',box_r)
    # return blank_image


def evaluate(model, loader, loss,epoch_number, args, prefix='', colors=None):
    errors = []
    model.eval()
    losses = None
    box_losses = []
    divergence_losses = []

    #for batch_i, (indexes, target) in tqdm(enumerate(loader)):
    length = len(loader)
    for batch_i, (indexes, target) in enumerate(loader):
        print('evaluate process: total: ',length, 'now: ',batch_i)
        label_set = torch.stack([t.label_set for t in target], dim=0).to(device)
        counts = torch.stack([t.count for t in target], dim=0).to(device)
        boxes = [t.bbox.to(device) for t in target]
        labels = [t.label.to(device) for t in target]
        number_boxes = np.stack([len(t) for t in target], axis=0)
        max_number_boxes = np.max(number_boxes)
        batch_size = label_set.size(0)

        predicted_boxes = torch.zeros((batch_size, max_number_boxes, 8)).to(device)
        # import ipdb; ipdb.set_trace()
        for step in range(max_number_boxes):
            # determine who has a box.
            has_box = number_boxes > step

            # determine their history of box/labels.
            current_label_set = label_set[has_box, :]
            current_counts = counts[has_box, :]

            all_boxes = [boxes[i] for i, has in enumerate(has_box) if has]
            all_labels = [labels[i] for i, has in enumerate(has_box) if has]
            current_label = torch.stack([l[step] for l in all_labels], dim=0).to(device)
            current_label = label_encodings[current_label.long() - 1]
            current_box = torch.stack([b[step] for b in all_boxes], dim=0).to(device)

            # now, consider the history.
            if step == 0:
                previous_labels = torch.zeros((batch_size, 0, 7)).to(device)
                previous_boxes = torch.zeros((batch_size, 0, 4)).to(device)
            else:
                previous_labels = torch.stack([l[step - 1] for l in all_labels], dim=0).unsqueeze(1)
                previous_labels = label_encodings[previous_labels.long() - 1]

                # we need to 1-hot these. only take the previous one since
                # we'll accumulate state instead.
                previous_boxes = torch.stack([b[step - 1] for b in all_boxes], dim=0).unsqueeze(1)

            # take a step. x, label_set, current_label, count_so_far):
            state = (h[has_box].unsqueeze(0), c[has_box].unsqueeze(0)) if step > 1 else None
            predicted_boxes_step, kl_divergence, z, state = model(current_box, current_label_set, current_label,
                                                                  previous_labels, previous_boxes, state=state)
            # print('predicted_boxes_step.shape',predicted_boxes_step.shape)
            # print('predicted_boxes[has_box, step].shape: ',predicted_boxes[has_box, step].shape)
            predicted_boxes[has_box, step] = predicted_boxes_step

            box_loss_step = loss(predicted_boxes_step, current_box)
            losses = box_loss_step if losses is None else torch.cat([losses, box_loss_step])

            box_losses.append(box_loss_step.reshape(-1))
            divergence_losses.append(kl_divergence.reshape(-1))

            if state is not None:
                h, c = torch.zeros((batch_size, 128)).to(device), torch.zeros((batch_size, 128)).to(device)
                h[has_box, :] = state[0][-1]
                c[has_box, :] = state[1][-1]

        # if batch_i == 0 and colors is not None:
        if colors is not None:
            # try plotting the first batch.
            for i in range(batch_size):
                count = number_boxes[i]
                # plotted = plot_layout(
                #     boxes[i].detach().cpu().numpy(),
                #     predicted_boxes[i, :count],
                #     labels[i].detach().cpu().numpy()-1,
                #     target[i].width,
                #     target[i].height,
                #     colors=colors)
                save_path =args.log_dir + '/' +f'images_{batch_i}'
                gt_txt_savepath = save_path+'/'+'boxes_txt'+'/'+str(i)
                boxes_np = boxes[i].detach().cpu().numpy()
                
                plot_layout(boxes_np,save_path+'/'+'gt',i)
                predict = predicted_boxes[i].detach().cpu().numpy()
                # predicted_txt_savepath = save_path+'/'+'predicted_boxes_txt'+'/'+str(i)
                plot_layout(predict*args.image_size,save_path+'/'+'predicted',i)
                # plotted.save(f"{prefix}_{i:05d}.png")

        # pdb.set_trace()
    average_loss = torch.mean(losses)
    print(f"validation: average loss: {average_loss}")
    count_losses = torch.cat(box_losses)
    divergence_losses = torch.cat(divergence_losses)
    loss_epoch = torch.mean(count_losses) + torch.mean(divergence_losses)

    return loss_epoch.item()


class GaussianLogLikelihood(nn.Module):
    def __init__(self):
        super(GaussianLogLikelihood, self).__init__()

        self.var = 0.02 ** 2

    def forward(self, predicted, expected):
        # not really sure if I am supposed to use the variance
        # stated in the paper.
        error = torch.mean((predicted - expected) ** 2, dim=-1)
        return error


class AutoregressiveBoxVariationalAutoencoder(nn.Module):
    def __init__(self, number_labels, conditioning_size, representation_size):
        super(AutoregressiveBoxVariationalAutoencoder, self).__init__()

        self.representation_size = representation_size

        self.encoder = AutoregressiveBoxEncoder(number_labels, conditioning_size, representation_size)
        self.decoder = AutoregressiveBoxDecoder(conditioning_size, representation_size)

    def sample(self, mu, log_var):
        batch_size = mu.size(0)
        device = mu.device

        standard_normal = torch.randn((batch_size, self.representation_size), device=device)
        z = mu + standard_normal * torch.exp(0.5 * log_var)

        kl_divergence = -0.5 * torch.sum(
            1 + log_var - (mu ** 2) - torch.exp(log_var), dim=1)

        return z, kl_divergence

    def forward(self, x, label_set, current_label, labels_so_far, boxes_so_far, state=None):
        # print('x.shape: ',x.shape)
        # print('label_set.shape: ',label_set.shape)
        # print('current_label.shape: ',current_label.shape)
        # print('labels_so_far.shape: ',labels_so_far.shape)
        # print('boxes_so_far.shape: ',boxes_so_far.shape)
        '''x.shape:  torch.Size([32, 5])
            label_set.shape:  torch.Size([32, 1])
            current_label.shape:  torch.Size([32, 1])
            labels_so_far.shape:  torch.Size([32, 1, 1])
            boxes_so_far.shape:  torch.Size([32, 1, 5])'''
        
        mu, s, condition, state = self.encoder(x, label_set, current_label, labels_so_far, boxes_so_far, state)
        flops_encoder, params_encoder = profile(self.encoder, (x, label_set, current_label, labels_so_far, boxes_so_far, state))
        logging.info(f'flops_encoder: {flops_encoder}')
        logging.info(f'params_encoder: {params_encoder}')

        # if torch.isnan(state).any():
        #     logging.info(f'state: {state}')
        z, kl_divergence = self.sample(mu, s)
        boxes = self.decoder(z, condition)
        flops_decoder, params_decoder = profile(self.decoder, (z, condition))
        logging.info(f'flops_decoder: {flops_decoder}')
        logging.info(f'params_decoder: {params_decoder}')
        return boxes, kl_divergence, z, state


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)
    # logging.basicConfig(filename=f'/home/shuhang/Desktop/paper_code/layouttransformer/DeepLayout-main/paramaters/model_paramaters.log', level=logging.INFO)

    parser = argparse.ArgumentParser('Box VAE')
    parser.add_argument("--exp", default="box_vae", help="postfix for experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--train_json", default="./instances_train.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="./instances_val.json", help="/path/to/val/json")

    parser.add_argument("--max_length", type=int, default=256, help="batch size")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9, help="beta_1 for adam")
    parser.add_argument('--evaluate', action='store_true',default=True, help="evaluate only")
    parser.add_argument('--save_every', type=int, default=10, help="evaluate only")
    parser.add_argument('--image_size', type=int, default=320, help="evaluate only")

    args = parser.parse_args()

    if not args.evaluate:
        now = datetime.now().strftime("%m%d%y_%H%M%S")
        log_dir = os.path.join(args.log_dir, f"{now}_{args.exp}")
        samples_dir = os.path.join(log_dir, "samples")
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        log_dir = args.log_dir
        samples_dir = os.path.join(log_dir, "samples")
        ckpt_dir = os.path.join(log_dir, "checkpoints")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    #save paramaters 
    logging_path = args.log_dir+'/'+'logging'
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logging.basicConfig(filename=f'{logging_path}/model_paramaters.log', level=logging.INFO)
    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')    
        
    collator = BatchCollator()
    train_dataset = LayoutDataset(args.train_json, args.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator)

    validation_dataset = LayoutDataset(args.val_json, args.max_length)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator)

    NUMBER_LABELS = train_dataset.number_labels
    colors = gen_colors(NUMBER_LABELS)

    label_encodings = torch.eye(NUMBER_LABELS).float().to(device)
    box_loss = GaussianLogLikelihood().to(device)

    autoencoder = AutoregressiveBoxVariationalAutoencoder(
        NUMBER_LABELS,
        conditioning_size=128,
        representation_size=32).to(device)

    # evaluate the model
    if args.evaluate:
        min_epoch = -1
        min_loss = 1e100
        for epoch in range(args.epochs):
            checkpoint_path = os.path.join(log_dir, "checkpoints", 'epoch_%d.pth' % epoch)
            checkpoint_path = '/scratch/sg7484/InfiniteCityGen/results/layout_vae/exp1/1e-5/090123_083210_box_coco_instances/checkpoints/epoch_153.pth'
            if not os.path.exists(checkpoint_path):
                continue
            print('Evaluating', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            autoencoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
            loss = evaluate(autoencoder, validation_loader, box_loss,epoch_number=epoch,args=args,colors = colors)
            print('End of epoch %d : %f' % (epoch, loss))
            if loss < min_loss:
                min_loss = loss
                min_epoch = epoch
        print('Best epoch: %d Best nll: %f' % (min_epoch, min_loss))
        sys.exit(0)

    # opt = optim.Adam(autoencoder.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
    # epoch_number = 0
    # writer = SummaryWriter(args.log_dir)
    # while True:
        

    #     if (epoch_number > 0) and (epoch_number == args.epochs):
    #         print("done!")
    #         break

    #     print(f"starting epoch {epoch_number+1}")
    #     autoencoder.train()
    #     print('epoch_number: ',epoch_number)
    #     # with tqdm(enumerate(train_loader)) as tq:
    #     length = len(train_loader)
    #     for batch_i, (indexes, target) in enumerate(train_loader):
    #         print('train process: total: ',length, 'now: ',batch_i)
    #         #print('batch_i: ',batch_i)
            
    #         autoencoder.zero_grad()
    #         box_loss.zero_grad()
    #         label_set = torch.stack([t.label_set for t in target], dim=0).to(device)
    #         counts = torch.stack([t.count for t in target], dim=0).to(device)
    #         boxes = [t.bbox.to(device) for t in target]
    #         labels = [t.label.to(device) for t in target]
    #         number_boxes = np.stack([len(t) for t in target], axis=0)
    #         max_number_boxes = np.max(number_boxes)

    #         batch_size = label_set.size(0)
    #         # previous_boxes = torch.zeros((batch_size, max_number_boxes, 4)).to(device)

    #         box_losses = []
    #         divergence_losses = []

    #         current_box_loss = torch.zeros((batch_size, max_number_boxes)).to(device)
    #         current_divergence_loss = torch.zeros((batch_size, max_number_boxes)).to(device)

    #         for step in range(max_number_boxes):

    #             # determine who has a box.
    #             has_box = number_boxes > step

    #             # determine their history of box/labels.
    #             current_label_set = label_set[has_box, :]
    #             current_counts = counts[has_box, :]

    #             all_boxes = [boxes[i] for i, has in enumerate(has_box) if has]
    #             all_labels = [labels[i] for i, has in enumerate(has_box) if has]
    #             current_label = torch.stack([l[step] for l in all_labels], dim=0).to(device)
    #             current_label = label_encodings[current_label.long() - 1]
    #             current_box = torch.stack([b[step] for b in all_boxes], dim=0).to(device)

    #             # now, consider the history.
    #             if step == 0:
    #                 previous_labels = torch.zeros((batch_size, 0, 7)).to(device)
    #                 previous_boxes = torch.zeros((batch_size, 0, 5)).to(device)
    #             else:
    #                 previous_labels = torch.stack([l[step - 1] for l in all_labels], dim=0).unsqueeze(1)
    #                 previous_labels = label_encodings[previous_labels.long() - 1]

    #                 # we need to 1-hot these. only take the previous one since
    #                 # we'll accumulate state instead.
    #                 previous_boxes = torch.stack([b[step - 1] for b in all_boxes], dim=0).unsqueeze(1)

    #             # take a step. x, label_set, current_label, count_so_far):
    #             state = (h[has_box].unsqueeze(0), c[has_box].unsqueeze(0)) if step > 1 else None
    #             predicted_boxes, kl_divergence, z, state = autoencoder(current_box, current_label_set, current_label,
    #                                                                     previous_labels, previous_boxes, state=state)
    #             if not (state is None):
    #                 h, c = torch.zeros((batch_size, 128)).to(device), torch.zeros((batch_size, 128)).to(device)
    #                 h[has_box, :] = state[0][-1]
    #                 c[has_box, :] = state[1][-1]

    #             box_loss_step = box_loss(predicted_boxes, current_box)

    #             current_box_loss[has_box, step] = box_loss_step
    #             current_divergence_loss[has_box, step] = kl_divergence

    #         number_boxes = torch.from_numpy(number_boxes).to(device).float()
    #         box_loss_batch = torch.mean(torch.sum(current_box_loss, dim=-1) / number_boxes)
    #         divergence_loss_batch = torch.mean(torch.sum(current_divergence_loss, dim=-1) / number_boxes)

    #         loss_batch = box_loss_batch + 0.0001 * divergence_loss_batch
    #         loss_batch.backward()
    #         opt.step()
            
    #         # tq.set_description(f"{epoch_number+1}/{args.epochs} box_loss: {box_loss_batch.item()}"
    #         #                     f"kl: {divergence_loss_batch.item()}")
    #         #print('loss_batch: ',loss_batch)
    #         writer.add_scalar('loss_train', loss_batch.item(), global_step=epoch_number)
    #         # logging.info(f'box_loss_batch.item(): {box_loss_batch.item()}')
    #         # logging.info(f'divergence_loss_batch.item(): {divergence_loss_batch.item()}')
    #         # logging.info(f'\n\n')
    #         if torch.isnan(box_loss_batch).any() or torch.isnan(divergence_loss_batch).any():
    #             sys.stderr.write('There is NaN\n')
    #             raise SystemExit(1)
            
    #     # if (epoch_number + 1) % 1 == 0:
    #     #   validation_loss, validation_accuracy = evaluate()
    #     #   print("validation loss [{0}/{1}: {2:4f}".format(epoch_number, NUMBER_EPOCHS, validation_loss.item()))
    #     #   # write out a checkpoint too.
    #     prefix = os.path.join(samples_dir, f"epoch_{epoch_number+1:03d}")
    #     test_loss = evaluate(autoencoder, validation_loader, box_loss, epoch_number,args = args, prefix=prefix, colors=colors)
    #     writer.add_scalar('test_loss', test_loss, global_step=epoch_number)
    #     torch.save({
    #         "epoch": epoch_number,
    #         "model_state_dict": autoencoder.state_dict(),
    #     }, os.path.join(ckpt_dir, "epoch_{0}.pth".format(epoch_number)))

    #     epoch_number += 1
