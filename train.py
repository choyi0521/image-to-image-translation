import argparse
from trainers.pix2pix_trainer import Pix2PixTrainer


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['pix2pix'], required=True)
    parser.add_argument('--dataset', type=str, default='./resources/facades')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--print_loss_freq', type=int, default=40)

    # data loader
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)

    # model hyperparameters
    parser.add_argument('--inner_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--lamb', type=float, default=10.0)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    args = parser.parse_args()

    trainer = None
    if args.model == 'pix2pix':
        trainer = Pix2PixTrainer(
            args=args
        )

    trainer.train()
    trainer.test()