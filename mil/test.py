import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import timm
from sklearn.metrics import accuracy_score, roc_auc_score
from PIL import Image
import torchvision.transforms as trasnforms
import argparse
import gc

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def main():
    # INIT ##############################
    parser = argparse.ArgumentParser(description="ViT finetuning")
    parser.add_argument("--dataset", type=str, default="../datasets/test", metavar="X",
                        help="dataset used for testing the model")
    
    parser.add_argument("--model", type=str, default="vit_base_patch16_384", metavar="X",
                        help="model name (from timm library)")
    
    parser.add_argument("--num_classes", type=int, default=2, metavar="N",
                        help="number of classes for the task")
    
    parser.add_argument("--pretrained", type=str, required=False, metavar="X",
                        help="path of the trained weights for the model, if not indicated : pretrained timm model used")
    
    args = parser.parse_args()

    torch.manual_seed(42)
    
    # LOAD ###############################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if use_cuda else "")

    folder_path = args.dataset
    folder_classes = ["Normal", "Tumor"]
    paths = [os.path.join(folder_path, folder_classes[0]), os.path.join(folder_path, folder_classes[1])]

    # define the transforms you want to apply
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
    ])
    
    # Model
    print(f"Create model {args.model} with {args.num_classes} classes", flush=True)
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
    if not (args.pretrained is None):
        print(f"Load trained weight from {args.pretrained}", flush=True)
        if use_cuda:
            checkpoint = torch.load(args.pretrained)
        else:
            checkpoint = torch.load(args.pretrained, map_location=(torch.device('cpu')))

        # load the state of the model from the 'blocks' module
        blocks_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        model.load_state_dict(blocks_state_dict, strict=False)

    model.to(device)
    model.eval()

    # TEST ###############################
    all_max_conf_patch = []
    all_mean_prediction = []
    all_max_prediction = []
    all_nb_tumor_prediction = []
    all_expected_values = []
    
    # Iterate over the 129 test WSIs
    with torch.no_grad():
        for img_nb in range(1,130):
            max_conf_patch = [{"patch":-1, "conf":0}, {"patch":-1, "conf":0}]
            mean_prediction = 0
            nb_tumor_prediction = 0

            if os.path.isfile(os.path.join(paths[0], f"test_{img_nb:03d}_1.png")):
                print(f"\ttest_{img_nb}: Normal")
                img_class = 0
            elif os.path.isfile(os.path.join(paths[1], f"test_{img_nb:03d}_1.png")):
                print(f"\ttest_{img_nb}: Tumor")
                img_class = 1
            else:
                print(f"\ttest_{img_nb}: Not found")
                continue
            
            
            # Iterate over 500 patches
            for patch_nb in range(0,500):
                filename = f"test_{img_nb:03d}_{patch_nb}.png"
                folder = os.path.join(folder_path, paths[img_class])
                # load the image
                img = Image.open(os.path.join(folder, filename))
                x = transform(img)
                x = x.unsqueeze(0)
                x.to(device)
                y = model(x) # Add batch size dimension (1)
                probs = F.softmax(y, dim=1)

                # Update mean for prediction 1 (with running mean)
                mean_prediction = (patch_nb * mean_prediction + probs[0, 1]) / (patch_nb + 1)

                # Update max confidence patches (for both classes)
                if max_conf_patch[0]["conf"] < probs[0, 0]:
                    max_conf_patch[0]["conf"] = probs[0, 0]
                    max_conf_patch[0]["patch"] = patch_nb
                
                if max_conf_patch[1]["conf"] < probs[0, 1]:
                    max_conf_patch[1]["conf"] = probs[0, 1]
                    max_conf_patch[1]["patch"] = patch_nb

                # Update number of class 1 prediction
                pred_class = torch.argmax(probs, dim=1)
                if pred_class == 1:
                    nb_tumor_prediction += 1

            
            # Choose between the two max conf values obtained (the most confidence patch from the most confidence patch for each patch )
            if max_conf_patch[0]["conf"] > max_conf_patch[1]["conf"]:
                all_max_prediction.append(1-max_conf_patch[0]["conf"].item())
            else:
                all_max_prediction.append(max_conf_patch[1]["conf"].item())


            all_expected_values.append(img_class)
            all_mean_prediction.append(mean_prediction.item())
            all_max_conf_patch.append(max_conf_patch)
            all_nb_tumor_prediction.append(nb_tumor_prediction)

            print(f"\tMean:{mean_prediction.item()}\tMax conf:{max_conf_patch}\tNb tumors:{nb_tumor_prediction}")

    print(f"Image pred (mean):\t{all_mean_prediction}")
    print(f"Image pred (max):\t{all_max_prediction}")
    print(f"Image labels:\t{all_expected_values}")
    print(f"Image nb tumors:\t{all_nb_tumor_prediction}")
    print(f"Image max conf:\t{all_max_conf_patch}")

    # Compute the patch-level accuracy and ROC AUC
    print(f"Compute scores", flush=True)
    max_patch_acc = accuracy_score(all_expected_values, [1 if x >= 0.5 else 0 for x in all_max_prediction])
    max_patch_auc = roc_auc_score(all_expected_values, all_max_prediction)

    # Compute the image-level accuracy and ROC AUC
    image_acc = accuracy_score(all_expected_values, [1 if x >= 0.5 else 0 for x in all_mean_prediction])
    image_auc = roc_auc_score(all_expected_values, all_mean_prediction)

    print(f"Image-level accuracy:\t{image_acc:.4f}")
    print(f"Image-level ROC AUC:\t{image_auc:.4f}")
    print(f"Image-level accuracy (max):\t{max_patch_acc:.4f}")
    print(f"Image-level ROC AUC (max):\t{max_patch_auc:.4f}")

if __name__ == "__main__":
    main()

