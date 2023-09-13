import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import ImageClassificationDataModule
from models_modules import ImageClassifier, FinetuningCallback
#from ploting import plot_losses
from codecarbon import EmissionsTracker

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="ViT finetuning")
    parser.add_argument("--dataset", type=str, default="HumanLba", metavar="N",
                        help="dataset used for the task")

    parser.add_argument("--output", type=str, default="./out", metavar="N",
                        help="output folder path")

    parser.add_argument("--pretrain", type=str, default="vit_base_patch16_224", metavar="N",
                        help="checkpoint of the pretrianed model")

    parser.add_argument("--checkpoint", type=str, default=None, metavar="N",
                        help="checkpoint of partly trained model ")

    parser.add_argument("--lr", type=float, default=0.0001, metavar="LR",
                        help="learning rate used")

    parser.add_argument("--start_lr", type=float, default=0.001, metavar="LR",
                        help="learning rate used at the start (when training only the classifier)")

    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="number of epochs")

    parser.add_argument("--start_epochs", type=int, default=0, metavar="N",
                        help="number of epochs when only the classifier is trained")

    parser.add_argument("--batch_size", type=int, default=32, metavar="N",
                        help="size of a batch")

    parser.add_argument("--use_roc_auc", type=bool, default=True, metavar="N",
                        help="if only 2 classes, decide if use roc auc or not")

    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Initialize CSVLogger
    ts_logger = TensorBoardLogger(save_dir=args.output[2:], name=args.pretrain, log_graph=False)

    # Initialize data module and model
    data_module = ImageClassificationDataModule(
        data_dir=os.path.join("./../../../datasets/", args.dataset),
        train_batch_size=args.batch_size,
        val_batch_size=100,
        image_size=224
    )
    data_module.setup()
    model = ImageClassifier(num_classes=len(data_module.train_dataset.classes), lr=args.start_lr, model_name=args.pretrain, pretrained=False)

    # Initialize trainer
    trainer = pl.Trainer(
        deterministic=True,
        gpus=(1 if torch.cuda.is_available() else None),  # Set to None to use CPU, or use an integer to specify the number of GPUs to use
        max_epochs=args.epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            FinetuningCallback(start_epochs = args.start_epochs, new_lr = args.lr)
        ],
        logger=ts_logger
    )

    trainer.logger._log_graph = False

    # Train the model
    tracker = EmissionsTracker()
    tracker.start()
    trainer.fit(model, datamodule=data_module)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")

    # Test the model
    trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=True, ckpt_path="best")

    # Save the model checkpoint
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    checkpoint_path = os.path.join(args.output, f"{args.dataset}_{args.pretrain}.ckpt")
    trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
        main()
