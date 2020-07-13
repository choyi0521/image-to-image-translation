import argparse
from core.torch_trainer import TorchTrainer

class Pix2PixTrainer(TorchTrainer):
    def __init__(self,
                 dataset,
                 load_dir,
                 save_dir
                 ):
        super().__init__()

    def train(self):
        print(self.device)
        pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['pix2pix'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--load_dir', type=str, required=False, default='./checkpoints')
    parser.add_argument('--save_dir', type=str, required=False, default='./checkpoints')
    args = parser.parse_args()

    trainer = None
    if args.model == 'pix2pix':
        trainer = Pix2PixTrainer(
            dataset=args.dataset,
            load_dir=args.load_dir,
            save_dir=args.save_dir
        )

    trainer.train()