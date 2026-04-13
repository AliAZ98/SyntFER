import warnings

warnings.filterwarnings("ignore")
# from apex import amp
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os
from tqdm import tqdm

import torch
import argparse
from data_preprocessing.dataset_raf import RafDataSet, RafGeneratedDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class

from sklearn.metrics import f1_score, confusion_matrix
from time import time
from utils import *
from data_preprocessing.sam import SAM
from models.emotion_hyp import pyramid_trans_expr
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from models.ir50 import Backbone as resnet50


import wandb 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000001095, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=120, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
    return parser.parse_args()


######################
class FER_IR50(nn.Module):
    def __init__(self, num_classes=7, drop_ratio=0.4, mode='ir', freeze_backbone=False):
        super().__init__()
        # Build backbone exactly as your code does
        self.backbone = resnet50(num_layers=50, drop_ratio=drop_ratio, mode=mode)

        # Classification head on top of backbone features
        # Your backbone forward returns (B, 49, 1024) = 7x7 spatial tokens of 1024-d
        # We'll global-average pool to (B, 1024) and map to 7 classes.
        self.head = nn.Linear(1024, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Backbone returns (B, 49, 1024)
        x = self.backbone(x)                    # (B, 49, 1024)
        B, N, C = x.shape                       # N should be 49, C=1024
        x1 = x.view(B, 7, 7, C).permute(0, 3, 1, 2)  # (B, 1024, 7, 7)
        x = F.adaptive_avg_pool2d(x1, 1).flatten(1)  # (B, 1024)
        logits = self.head(x)                        # (B, 7)
        return logits , x1

def smart_load_backbone(model, ckpt_path, device="cpu"):
    import torch
    sd_model = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location=device)
    src = ckpt.get("state_dict", ckpt)

    def strip_prefix(k):
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        return k

    filtered = {}
    kept, skipped = 0, 0
    for k, v in src.items():
        k0 = strip_prefix(k)                    # e.g. "input_layer.0.weight"
        target_k = "backbone." + k0             # we load into model.backbone.*
        if target_k in sd_model and sd_model[target_k].shape == v.shape:
            filtered[target_k] = v
            kept += 1
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[smart_load_backbone] kept={kept}, skipped={skipped}")
    print("[smart_load_backbone] missing (in model, not in ckpt):", missing)
    print("[smart_load_backbone] unexpected (in ckpt, not used):", unexpected)
    return model

def run_training():
    args = parse_args()
    torch.manual_seed(1376)

    dataset_type = "mixed_syntface_Wrafdb"
    wandbname = dataset_type + '1'
    ### WANDB LOGIN ###
    wandb.login(key="331aa21eaee256e47f5a11eb3afe391a45729ec3")
    wandb.init(project="Synthetic_FER_project_ir50", name=f"{wandbname}", config=args,entity="simit")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])



    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
        #v2.GaussianNoise(mean= 0.0, sigma= 0.3, clip=True),
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    if args.dataset == "rafdb":
        num_classes = 7
        # Root will be given inside the dataset class
        train_dataset = Affectdataset(root=None, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset(root=None, train=False, transform=data_transforms_val)
        model = FER_IR50(num_classes=7, drop_ratio=0.4, mode='ir', freeze_backbone=False)
        # Path to pretrained backbone weights
        ckpt_path = "/path/to/pretrained/path/ir50.pth"
        # Load pretrained backbone weights into the model
        smart_load_backbone(model, ckpt_path, device="cpu")
    elif args.dataset == "affectnet":
        datapath = '/path/to/FER/raf_db/dataset/'
        num_classes = 7
        train_dataset = Affectdataset(root=None, train=True, transform=data_transforms, basic_aug=True)
        #val_dataset = Affectdataset(root=None, train=False, transform=data_transforms_val)
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        train_dataset = Affectdataset_8class(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')

    
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # model = Networks.ResNet18_ARM___RAF()

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    params = model.parameters()
    if args.optimizer == 'adamw':
        # base_optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.AdamW(weight_decay=1e-4)
    elif args.optimizer == 'adam':
        # base_optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.Adam#(weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        # base_optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")
    # print(optimizer)
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, weight_decay=1e-4)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)   

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.4)


    best_acc = 0

    from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

    # Initialize metrics
    train_accuracy_metric = Accuracy(task='multiclass',average='macro', num_classes=num_classes).cuda()
    train_precision_metric = Precision(task='multiclass',average='macro', num_classes=num_classes).cuda()
    train_recall_metric = Recall(task='multiclass',average='macro', num_classes=num_classes).cuda()
    train_f1_metric = F1Score(task='multiclass',average='macro', num_classes=num_classes).cuda()

    val_accuracy_metric = Accuracy(task='multiclass',average='macro', num_classes=num_classes).cuda()
    val_precision_metric = Precision(task='multiclass',average='macro', num_classes=num_classes).cuda()
    val_recall_metric = Recall(task='multiclass',average='macro', num_classes=num_classes).cuda()
    val_f1_metric = F1Score(task='multiclass',average='macro', num_classes=num_classes).cuda()

    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        # Reset metrics for training
        train_accuracy_metric.reset()
        train_precision_metric.reset()
        train_recall_metric.reset()
        train_f1_metric.reset()
        # Initialize tqdm for the training loop
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {i} - Training", leave=False)
        
        for batch_i, (imgs, targets) in train_bar:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, features = model(imgs)
            targets = targets.cuda()

            # Compute loss
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            outputs, features = model(imgs)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)

            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.second_step(zero_grad=True)

            train_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            
            train_accuracy_metric.update(predicts, targets)
            train_precision_metric.update(predicts, targets)
            train_recall_metric.update(predicts, targets)
            train_f1_metric.update(predicts, targets)

            # Update progress bar
            current_loss = train_loss / iter_cnt
            current_acc = train_accuracy_metric.compute().item()
            train_bar.set_postfix({"Loss": current_loss, "Accuracy": current_acc, "Precision": train_precision_metric.compute().item(), "Recall": train_recall_metric.compute().item(), "F1": train_f1_metric.compute().item()})


        
        train_acc = train_accuracy_metric.compute().item()
        train_precision = train_precision_metric.compute().item()
        train_recall = train_recall_metric.compute().item()
        train_f1 = train_f1_metric.compute().item()
        train_loss /= iter_cnt
        elapsed = (time() - start_time) / 60

        wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "epoch": i
                    })
        print(f'[Epoch {i}] Train time: {elapsed:.2f} min, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Loss: {train_loss:.3f}')

        scheduler.step()


        class_names = ['anger', 'disgust', 'fear', 'happy', 'Neutral', 'sad', 'surprise']

        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            # Reset metrics for validation
            val_accuracy_metric.reset()
            val_precision_metric.reset()
            val_recall_metric.reset()
            val_f1_metric.reset()
            # Initialize tqdm for the validation loop
            val_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {i} - Validation", leave=False)
            all_targets = []
            all_predictions = []
            for batch_i, (imgs, targets) in val_bar:
                outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                # Compute loss
                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)


                all_predictions.extend(predicts.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                val_accuracy_metric.update(predicts, targets)
                val_precision_metric.update(predicts, targets)
                val_recall_metric.update(predicts, targets)
                val_f1_metric.update(predicts, targets)

                # Update the progress bar with current loss and accuracy
                current_val_loss = val_loss / iter_cnt
                current_val_acc = val_accuracy_metric.compute()

                val_bar.set_postfix({"Val Loss": current_val_loss, "Val Accuracy": current_val_acc.item(), "Val Precision": val_precision_metric.compute().item(), "Val Recall": val_recall_metric.compute().item(), "Val F1": val_f1_metric.compute().item()})

            val_loss = val_loss / iter_cnt
            val_acc = val_accuracy_metric.compute().item()
            val_precision = val_precision_metric.compute().item()
            val_recall = val_recall_metric.compute().item()
            val_f1 = val_f1_metric.compute().item()
            total_score = 0.67 * val_f1 + 0.33 * val_acc

            cm = confusion_matrix(all_targets, all_predictions, labels=list(range(len(class_names))))
            print("Confusion Matrix:")
            print(cm)

            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.title(f"Confusion Matrix for Epoch {i}")
            plt.xticks(rotation=45)

            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_total_score": total_score,
                "epoch": i,
                "confusion_matrix": wandb.Image(fig)
            })

            save_dir = f'/path/to/checkpoints_of_ir50/{dataset_type}_Checkpoints'
            os.makedirs(save_dir, exist_ok=True)  # Create if not exists
            if i % 10 == 0:
                best_acc = 0
            if total_score > best_acc:  # val_acc > 0.60 and
                torch.save(
                    {
                        'iter': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    os.path.join(save_dir, f"epoch{i}_acc{val_acc}.pth")
                )
                print('Model saved.')

            if total_score > best_acc:
                best_acc = total_score
                print(f"!!!NEW Best accuracy: {best_acc}!!!")
    print('Finished Training and the best accuracy is: ', best_acc)
    wandb.finish()


if __name__ == "__main__":
    run_training()
