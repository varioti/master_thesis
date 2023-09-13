import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import timm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from PIL import Image
import argparse
import random
from torchvision import transforms
from codecarbon import EmissionsTracker
import tensorflow as tf

def main():
    # INIT ##############################
    parser = argparse.ArgumentParser(description="ViT finetuning")
    parser.add_argument("--dataset", type=str, required=False, metavar="X",
                        help="dataset used for training the model")
    
    parser.add_argument("--val_dataset", type=str, required=False, metavar="X",
                        help="dataset used for validating the model")
    
    parser.add_argument("--test_dataset", type=str, required=False, metavar="X",
                        help="dataset used for testing the model")
    
    parser.add_argument("--model", type=str, default="vit_base_patch16_384", metavar="X",
                        help="model name (from timm library)")
    
    parser.add_argument("--output", type=str, default="out", metavar="X",
                        help="output folder (to save ckpt weights)")

    parser.add_argument("--epochs", type=int, default=50, metavar="N",
                        help="number of epochs to train the model")
    
    parser.add_argument("--lr", type=float, default=1e-5, metavar="N",
                        help="learning rate used")
    
    parser.add_argument("--start_lr", type=float, default=1e-3, metavar="N",
                        help="starting learning rate used (when all layers expect last one is freezed)")
    
    parser.add_argument("--start_epochs", type=float, default=0, metavar="N",
                        help="number of epochs in which all layers are freezed (except las one)")

    parser.add_argument("--augment", type=bool, default=False, metavar="True",
                        help="if True: data augmentation")
    
    parser.add_argument("--topk", type=int, default=1, metavar="N",
                        help="top k tiles are assumed to be of the same class as the slide")
    
    args = parser.parse_args()

    torch.manual_seed(42)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if use_cuda else "")

    # LOAD ###############################
    train_folder_path = args.dataset
    val_folder_path = args.val_dataset
    test_folder_path = args.test_dataset
    folder_classes = ["Normal", "Tumor"]

    train_WSIs = []
    train_labels = []
    val_WSIs = []
    val_labels = []
    test_WSIs = []
    test_labels = []

    # For each dataset class, load the WSI names
    for i in range(len(folder_classes)):
        # Load validation folder only if specified
        if train_folder_path is not None:
            train_class_wsis = list(set([f"{wsi.split('_')[0]}_{wsi.split('_')[1]}" for wsi in os.listdir(os.path.join(train_folder_path, folder_classes[i])) if wsi.endswith('_0.png')]))
            train_class_labels = [i for _ in train_class_wsis]

            train_WSIs += train_class_wsis
            train_labels += train_class_labels

        # Load validation folder only if specified
        if val_folder_path is not None:
            val_class_wsis = list(set([f"{wsi.split('_')[0]}_{wsi.split('_')[1]}" for wsi in os.listdir(os.path.join(val_folder_path, folder_classes[i])) if wsi.endswith('_0.png')]))
            val_class_labels = [i for _ in val_class_wsis]

            val_WSIs += val_class_wsis
            val_labels += val_class_labels

        # Load validation folder only if specified
        if test_folder_path is not None:
            test_class_wsis = list(set([f"{wsi.split('_')[0]}_{wsi.split('_')[1]}" for wsi in os.listdir(os.path.join(test_folder_path, folder_classes[i])) if wsi.endswith('_0.png')]))
            test_class_labels = [i for _ in test_class_wsis]

            test_WSIs += test_class_wsis
            test_labels += test_class_labels

    if train_folder_path is not None:
        #Shuffle the training set and keep track of correct label for each WSI
        combined = list(zip(train_WSIs, train_labels))
        random.shuffle(combined)
        train_WSIs, train_labels = zip(*combined)

    print("Datasets loaded", flush=True)
    print(f"Train: {len(train_WSIs)} WSIs", flush=True)
    print(f"Valid: {len(val_WSIs)} WSIs", flush=True)
    print(f"Test : {len(test_WSIs)} WSIs", flush=True)

    # Define the transforms applied (training)
    image_size = 384
    augment_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Define the transforms applied (training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
    ])
    
    # Model
    print(f"Create model {args.model} with 2 classes", flush=True)
    model = timm.create_model(args.model, pretrained=True, num_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.start_lr)
    tracker = EmissionsTracker()

    # Freeze all layers except the last one
    print("Freeze layers except head")
    for name, param in model.named_parameters():
        if not name.startswith('head'):
            param.requires_grad = False

    # create a summary writer
    tf_writer = tf.summary.create_file_writer(os.path.join(args.output, f"{args.model}_log.0"))

    # Train ###############################
    if train_folder_path is not None:
        best_val_loss = float('inf')
        best_epoch = 0
        tracker.start()

        for epoch in range(args.epochs):
            if epoch == args.start_epochs:
                # Unfreeze all layers except the last one
                for name, param in model.named_parameters():
                        param.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            
            # Start training epoch
            data = []
            labels = []
            for wsi_idx in range(len(train_WSIs)):
                wsi_name = train_WSIs[wsi_idx]
                wsi_label = train_labels[wsi_idx]
                wsi_folder = os.path.join(train_folder_path, folder_classes[wsi_label])
                
                # Make inference for each tile of the wsi and select the k most big predictions of being positive (class 1)
                tr = augment_transform if args.augment else transform
                max_tiles, _ = select_tiles(epoch, model, device, tr, wsi_folder, wsi_name, args.topk)
                data += max_tiles
                labels += [wsi_label] * len(max_tiles)

            # Train model on the selected tiles
            loss = train(data, labels, model, device, criterion, optimizer)
            print(f"Epoch {epoch+1}/{args.epochs}: Loss = {loss:.4f}", flush=True)
            with tf_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)

            # Validation
            if val_folder_path is not None:
                print(f"Epoch {epoch+1}/{args.epochs}: Validation", flush=True)
                model.eval()
                num_correct = 0
                total_loss = 0
                with torch.no_grad():
                    for wsi_idx in range(len(val_WSIs)):
                        wsi_name = val_WSIs[wsi_idx]
                        wsi_label = val_labels[wsi_idx]
                        wsi_folder = os.path.join(val_folder_path, folder_classes[wsi_label])
                        _, max_preds = select_tiles(epoch, model, device, transform, wsi_folder, wsi_name, 1)
                        wsi_pred = max_preds[0]
                        wsi_loss = criterion(torch.tensor([[1-wsi_pred, wsi_pred]]), torch.tensor([wsi_label]))
                        total_loss += wsi_loss.item()
                        if round(wsi_pred) == wsi_label:
                            num_correct += 1
                
                accuracy = num_correct / len(val_WSIs)
                validation_loss = total_loss / len(val_WSIs)
                print(f"\tValidation Loss: {validation_loss}, Accuracy: {accuracy}", flush=True)
                with tf_writer.as_default():
                    tf.summary.scalar('val_loss', validation_loss, step=epoch)
                    tf.summary.scalar('val_acc', accuracy, step=epoch)

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_epoch = epoch+1
                    obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_loss': best_val_loss,
                        'acc': accuracy,
                        'optimizer' : optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(args.output, f"MIL_{args.model}.pth"))
                    print(f"\tChekpoint replaced, new best val loss = {best_val_loss} (epoch {epoch+1})", flush=True)
        
        
        print(f"Best epoch:{best_epoch} (val loss = {best_val_loss})")
        # Stop measuring carbon emissions
        print(f"{best_val_loss}")
        emissions: float = tracker.stop()
        print(f"Emissions: {emissions} kg")
    
    # Test ###############################
    if test_folder_path is not None:
        print(f"\nTesting (load best model at MIL_{args.model}.pth):", flush=True)
        best_ckpt =torch.load(os.path.join(args.output, f"MIL_{args.model}.pth"))
        model.load_state_dict(best_ckpt['state_dict'])
        print(f"Load OK", flush=True)
        model.eval()
        total_loss = 0
        predictions = []
        with torch.no_grad():
            for wsi_idx in range(len(test_WSIs)):
                wsi_name = test_WSIs[wsi_idx]
                wsi_label = test_labels[wsi_idx]
                wsi_folder = os.path.join(test_folder_path, folder_classes[wsi_label])
                _, max_preds = select_tiles(0, model, device, transform, wsi_folder, wsi_name, 1)
                wsi_pred = max_preds[0]
                wsi_loss = criterion(torch.tensor([[1-wsi_pred, wsi_pred]]), torch.tensor([wsi_label]))
                total_loss += wsi_loss.item()
                predictions.append(wsi_pred)
        
        # Compute the accuracy and ROC AUC on testing set
        bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]

        image_acc = accuracy_score(test_labels, bin_predictions)
        image_auc = roc_auc_score(test_labels, predictions)
        image_cm = confusion_matrix(test_labels, bin_predictions)
        image_loss = total_loss / len(test_WSIs)
        
        print(test_labels)
        print(predictions)
        print(f"Test WSI Loss    :\t{image_loss:.4f}", flush=True)
        print(f"Test WSI Accuracy:\t{image_acc:.4f}", flush=True)
        print(f"Test WSI ROC AUC :\t{image_auc:.4f}", flush=True)
        print(f"Test WSI Conf Mat:\n{image_cm}", flush=True)


def select_tiles(epoch, model, device, transform, wsi_folder, wsi_name, k):
    # Use the model to predict each tile output
    predictions = []
    tiles = []
    with torch.no_grad():
        model.eval()
        for tile_idx in range(0,3): 
            tile_path = os.path.join(wsi_folder, f"{wsi_name}_{tile_idx}.png")
            tile = Image.open(tile_path)
            tile = transform(tile)
            tiles.append(tile)
            pred = model(tile.unsqueeze(0).to(device))
            predictions.append(F.softmax(pred, dim=1)[0][1].item())

    # Select the k tiles with highest prediction scores
    indices = np.argsort(predictions)[::-1][:k]
    max_tiles = [tiles[i] for i in indices]
    max_preds = [predictions[i] for i in indices]
    return max_tiles, max_preds


def train(data, labels, model, device, criterion, optimizer, batch_size=16):
    model.train()
    running_loss = 0.0
    for i in range(0, len(data), batch_size):
        inputs = torch.stack(data[i:i+batch_size]).to(device)
        labels_batch = torch.tensor(labels[i:i+batch_size]).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data)

if __name__ == '__main__':
    main()
