import argparse
from trainers.pix2pix_trainer import Pix2PixTrainer


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['pix2pix'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--load_dir', type=str, required=False, default='./checkpoints')
    parser.add_argument('--save_dir', type=str, required=False, default='./checkpoints')

    # data loader
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

    # model hyperparameters
    parser.add_argument('--inner_channels', type=int, default=64)

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    trainer = None
    if args.model == 'pix2pix':
        trainer = Pix2PixTrainer(
            args=args
        )

    trainer.train()