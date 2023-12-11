import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json
from torch.utils.data.dataloader import DataLoader
import cv2
from utils import trim_tokens, gen_colors
import os
import matplotlib.pyplot as plt



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

        if batch_i == 0 and colors is not None:
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
                save_path =args.log_dir + '/' +f'images_{epoch_number}'
                plot_layout(boxes[i].detach().cpu().numpy(),save_path+'/'+'boxes',i)
                predict = predicted_boxes[i].detach().cpu().numpy()
                plot_layout(predict*args.image_size,save_path+'/'+'predicted_boxes',i)
                # plotted.save(f"{prefix}_{i:05d}.png")

        # pdb.set_trace()
    average_loss = torch.mean(losses)
    print(f"validation: average loss: {average_loss}")
    count_losses = torch.cat(box_losses)
    divergence_losses = torch.cat(divergence_losses)
    loss_epoch = torch.mean(count_losses) + torch.mean(divergence_losses)

    return loss_epoch.item()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Box VAE')
    parser.add_argument("--exp", default="box_vae", help="postfix for experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--train_json", default="./instances_train.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="./instances_val.json", help="/path/to/val/json")

    parser.add_argument("--max_length", type=int, default=256, help="batch size")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9, help="beta_1 for adam")
    parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--save_every', type=int, default=10, help="evaluate only")
    parser.add_argument('--image_size', type=int, default=320, help="evaluate only")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = BatchCollator()
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