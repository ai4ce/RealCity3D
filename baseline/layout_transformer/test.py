import os
import argparse
import torch
from dataset_test import MNISTLayout, JSONLayout
from model import GPT, GPTConfig
from trainer_test import Trainer, TrainerConfig
from utils import set_seed
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")

    # MNIST options
    parser.add_argument("--data_dir", default=None, help="/path/to/mnist/data")
    parser.add_argument("--threshold", type=int, default=16, help="threshold for grayscale values")

    # COCO/PubLayNet options
    # parser.add_argument("--train_json", default="./instances_train.json", help="/path/to/train/json")
    # parser.add_argument("--val_json", default="./instances_val.json", help="/path/to/val/json")
    parser.add_argument("--train_data", default="./instances_val.json", help="/path/to/val/json")
    parser.add_argument("--test_data", default="./instances_val.json", help="/path/to/val/json")

    # Layout options
    parser.add_argument("--max_length", type=int, default=161, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")
    parser.add_argument('--model_path', type=str, default='/scratch/sg7484/InfiniteCityGen/results/layout_transformer/data_xyhw_rad_8/n_layer_8-4.5e_06/publaynet/checkpoints/checkpoint_33.pth', help="trained model")

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    #save paramaters 
    logging_path = args.log_dir+'/'+'logging'
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logging.basicConfig(filename=f'{logging_path}/model_paramaters.log', level=logging.INFO)
    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')    
    
    # MNIST Testing
    if args.data_dir is not None:
        train_dataset = MNISTLayout(args.log_dir, train=True, threshold=args.threshold)
        valid_dataset = MNISTLayout(args.log_dir, train=False, threshold=args.threshold,
                                    max_length=train_dataset.max_length)
    # COCO and PubLayNet
    else:
        train_dataset = JSONLayout(args.train_data,max_length =args.max_length)
        valid_dataset = JSONLayout(args.test_data, max_length=train_dataset.max_length)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
    model = GPT(mconf)
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    trainer.train()
