 
"""ITCS 6169 Assignment 2: supervised baseline and SimCLR from scratch.

Run in order:
 python assignment2_ssl.py --stage prepare
 python assignment2_ssl.py --stage baseline
 python assignment2_ssl.py --stage ssl
 python assignment2_ssl.py --stage linear
 python assignment2_ssl.py --stage finetune
 python assignment2_ssl.py --stage evaluate --model finetune
 python assignment2_ssl.py --stage predict --model finetune

Requires: torch torchvision numpy pillow tqdm
All training uses the extracted train/ folder only. Validation is held out from training and SSL.
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare(root):
    """Validate already-extracted train/, test/, and test2/ folders."""
    for name in ('train', 'test', 'test2'):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing dataset folder: {folder}')
        images = [p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not images:
            raise ValueError(f'No image files found in {folder}')
        print(f'{name}: {len(images)} images found in {folder}')
    print('Dataset folders verified. No ZIP files required.')

def split_indices(ds, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(len(ds.classes)):
        ids = np.flatnonzero(np.asarray(ds.targets) == label)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction))
        val.extend(ids[:n_val].tolist()); train.extend(ids[n_val:].tolist())
    return train, val

class TwoViews(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        return im, im  # ImageFolder transform is a stochastic TwoView transform; split below

class ContrastiveTransform:
    def __init__(self, size):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=.25, contrast=.25)], p=.7),
            transforms.RandomGrayscale(p=.15),
            transforms.ToTensor(),
            transforms.Normalize([.5]*3, [.5]*3),
        ])
    def __call__(self, image):
        image = image.convert('RGB')
        return self.aug(image), self.aug(image)

# def loaders(args, mode):   Used during SimCLR + Linear probe, SimCLR + full fine-tuning and SimCLR + regularized fine-tuning
#     base = args.root / 'train'
#     plain = datasets.ImageFolder(base)
#     train_ids, val_ids = split_indices(plain, args.val_fraction, args.seed)
#     norm = transforms.Normalize([.5]*3, [.5]*3)
#     train_aug = transforms.Compose([
#         transforms.RandomResizedCrop(args.size, scale=(.75,1.0)),
#         transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
#     eval_aug = transforms.Compose([transforms.Resize((args.size,args.size)), transforms.ToTensor(), norm])
#     if mode == 'ssl':
#         train_ds = Subset(datasets.ImageFolder(base, transform=ContrastiveTransform(args.size)), train_ids)
#     else:
#         train_ds = Subset(datasets.ImageFolder(base, transform=train_aug), train_ids)
#     val_ds = Subset(datasets.ImageFolder(base, transform=eval_aug), val_ids)
#     opts = dict(batch_size=args.batch_size, num_workers=args.workers,
#                 pin_memory=torch.cuda.is_available())
#     return (DataLoader(train_ds, shuffle=True, drop_last=(mode=='ssl'), **opts),
#             DataLoader(val_ds, shuffle=False, **opts), plain.classes)


def loaders(args, mode):
    base = args.root / 'train'
    plain = datasets.ImageFolder(base)

    train_ids, val_ids = split_indices(
        plain, args.val_fraction, args.seed
    )

    norm = transforms.Normalize([.5] * 3, [.5] * 3)

    # Experiment 5: stronger supervised augmentation
    if args.stage == 'finetune_v3':
        train_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                args.size,
                scale=(0.65, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=0.5),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.12)
            )
        ])
    else:
        # Original augmentation for Experiments 1-4
        train_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                args.size, scale=(0.75, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ])

    eval_aug = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        norm
    ])

    if mode == 'ssl':
        train_ds = Subset(
            datasets.ImageFolder(
                base,
                transform=ContrastiveTransform(args.size)
            ),
            train_ids
        )
    else:
        train_ds = Subset(
            datasets.ImageFolder(base, transform=train_aug),
            train_ids
        )

    val_ds = Subset(
        datasets.ImageFolder(base, transform=eval_aug),
        val_ids
    )

    opts = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    return (
        DataLoader(
            train_ds,
            shuffle=True,
            drop_last=(mode == 'ssl'),
            **opts
        ),
        DataLoader(
            val_ds,
            shuffle=False,
            **opts
        ),
        plain.classes
    )

def encoder():
    # weights=None is essential: no ImageNet or other external pretrained weights.
    net = models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = encoder()
        self.projector = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512,128))
    def forward(self,x): return self.projector(self.backbone(x))

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__(); self.backbone=encoder(); self.fc=nn.Linear(512,n_classes)
    def forward(self,x): return self.fc(self.backbone(x))

def ntxent(z1,z2,temp):
    z = nn.functional.normalize(torch.cat([z1,z2],0),dim=1)
    n=z1.size(0)
    logits=(z @ z.T)/temp
    logits.fill_diagonal_(-1e9)
    targets=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return nn.functional.cross_entropy(logits,targets)

# def train_supervised(model, tr, va, args, filename, freeze=False, epochs=None):
#     device=args.device; model.to(device)
#     if freeze:
#         for p in model.backbone.parameters(): p.requires_grad=False
#     optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=1e-4)
#     epochs=epochs or args.epochs
#     scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
#     criterion=nn.CrossEntropyLoss()
#     best=-1; history=[]
#     for epoch in range(epochs):
#         model.train()
#         if freeze: model.backbone.eval()
#         total=correct=0; loss_sum=0.
#         for x,y in tqdm(tr,desc=f'{filename} epoch {epoch+1}/{epochs}',leave=False):
#             x,y=x.to(device),y.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             logits=model(x); loss=criterion(logits,y)
#             loss.backward(); optimizer.step()
#             total+=len(y); correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
#         scheduler.step()
#         val_acc,val_loss=score(model,va,device)
#         row=dict(epoch=epoch+1,train_acc=correct/total,val_acc=val_acc,train_loss=loss_sum/total,val_loss=val_loss)
#         history.append(row); print(row)
#         if val_acc>best:
#             best=val_acc
#             torch.save({'model':model.state_dict(),'classes':args.classes,'size':args.size,'val_acc':best},args.output/(filename+'.pt'))
#     write_csv(args.output/(filename+'_history.csv'),history)
#     print('Best validation accuracy:',round(best*100,2),'%')


def train_supervised(model, tr, va, args, filename, freeze=False, epochs=None):
    device = args.device
    model.to(device)

    if freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False

    regularized = filename in ('finetune_v2', 'finetune_v3', 'finetune_v4')
    use_mixup = filename == 'finetune_v3'

    if regularized:
        optimizer = torch.optim.AdamW(
            [
                {
                    'params': model.backbone.parameters(),
                    'lr': args.lr * args.encoder_lr_factor
                },
                {
                    'params': (
                        p for name, p in model.named_parameters()
                        if not name.startswith('backbone.')
                    ),
                    'lr': args.lr
                }
            ],
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

    epochs = epochs or args.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best = -1
    history = []

    for epoch in range(epochs):
        model.train()

        if freeze:
            model.backbone.eval()

        total = correct = 0
        loss_sum = 0.0

        for x, y in tqdm(
            tr,
            desc=f'{filename} epoch {epoch+1}/{epochs}',
            leave=False
        ):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_mixup:
                lam = float(
                    torch.distributions.Beta(0.2, 0.2).sample()
                )

                indices = torch.randperm(
                    x.size(0), device=device
                )

                mixed_x = (
                    lam * x +
                    (1.0 - lam) * x[indices]
                )

                logits = model(mixed_x)

                loss = (
                    lam * criterion(logits, y) +
                    (1.0 - lam) *
                    criterion(logits, y[indices])
                )

                # Accuracy against the unmixed labels is not
                # meaningful when MixUp is active.
                # Track weighted label agreement instead.
                predictions = logits.argmax(1)
                correct += (
                    lam * (predictions == y).sum().item()
                    + (1.0 - lam) *
                    (predictions == y[indices]).sum().item()
                )
            else:
                logits = model(x)
                loss = criterion(logits, y)

                correct += (
                    logits.argmax(1) == y
                ).sum().item()

            loss.backward()
            optimizer.step()

            total += len(y)
            loss_sum += loss.item() * len(y)

        scheduler.step()

        val_acc, val_loss = score(model, va, device)

        row = dict(
            epoch=epoch + 1,
            train_acc=correct / total,
            val_acc=val_acc,
            train_loss=loss_sum / total,
            val_loss=val_loss
        )

        history.append(row)
        print(row)

        if val_acc > best:
            best = val_acc

            torch.save(
                {
                    'model': model.state_dict(),
                    'classes': args.classes,
                    'size': args.size,
                    'val_acc': best
                },
                args.output / (filename + '.pt')
            )

    write_csv(
        args.output / (filename + '_history.csv'),
        history
    )

    print(
        'Best validation accuracy:',
        round(best * 100, 2),
        '%'
    )



def score(model,loader,device):
    model.eval(); n=right=0; total_loss=0.
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x);right+=(out.argmax(1)==y).sum().item();n+=len(y)
            total_loss+=nn.functional.cross_entropy(out,y,reduction='sum').item()
    return right/n,total_loss/n

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run_ssl(args,tr):
    model=SimCLR().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.ssl_lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.ssl_epochs)
    history=[]
    for ep in range(args.ssl_epochs):
        model.train(); total=0; n=0
        for (v1,v2),_ in tqdm(tr,desc=f'SSL epoch {ep+1}/{args.ssl_epochs}',leave=False):
            v1,v2=v1.to(args.device),v2.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss=ntxent(model(v1),model(v2),args.temperature)
            loss.backward();opt.step();total+=loss.item()*len(v1);n+=len(v1)
        sched.step();row={'epoch':ep+1,'ssl_loss':total/n};history.append(row);print(row)
        torch.save({'backbone':model.backbone.state_dict(),'size':args.size},args.output/'ssl_encoder.pt')
    write_csv(args.output/'ssl_history.csv',history)

def load_classifier(args,name):
    p=args.output/(name+'.pt')
    ckpt=torch.load(p,map_location='cpu',weights_only=True)
    model=Classifier(len(ckpt['classes']))
    model.load_state_dict(ckpt['model']);model.to(args.device)
    return model,ckpt

def evaluate(args):
    model,ckpt=load_classifier(args,args.model)
    t=transforms.Compose([transforms.Resize((ckpt['size'],ckpt['size'])),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    ds=datasets.ImageFolder(args.root/'test',transform=t)
    if ds.classes!=ckpt['classes']:raise ValueError('Test class order differs from training')
    loader=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers)
    acc,loss=score(model,loader,args.device)
    print(f'{args.model} TEST accuracy: {acc*100:.2f}% ({round(acc*len(ds))}/{len(ds)}), loss {loss:.4f}')
    return acc

def predict(args):
    model,ckpt=load_classifier(args,args.model);model.eval()
    t=transforms.Compose([transforms.Resize((ckpt['size'],ckpt['size'])),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    paths=sorted((args.root/'test2').glob('*'),key=lambda p:int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else p.name)
    rows=[]
    with torch.no_grad():
        for p in tqdm(paths,desc='Predict'):
            if p.suffix.lower() not in ('.jpg','.png','.jpeg'):continue
            with Image.open(p) as im: x=t(im.convert('RGB')).unsqueeze(0).to(args.device)
            logits=model(x); probs=logits.softmax(1)[0]; idx=int(probs.argmax())
            rows.append({'filename':p.name,'predicted_class':ckpt['classes'][idx], 'confidence':round(float(probs[idx]),6)})
    write_csv(args.output/(args.model+'_test2_predictions.csv'),rows)
    print('Wrote',len(rows),'predictions')


def analyze_validation(args):
    import csv

    # Load the best high-resolution model.
    model, ckpt = load_classifier(args, 'finetune_v4')
    model.eval()

    classes = ckpt['classes']
    num_classes = len(classes)

    # Reconstruct the exact validation split.
    base = args.root / 'train'
    plain = datasets.ImageFolder(base)

    if plain.classes != classes:
        raise ValueError(
            'Training class order differs from checkpoint.'
        )

    _, val_ids = split_indices(
        plain,
        args.val_fraction,
        args.seed
    )

    # Match the evaluation preprocessing used during training.
    transform = transforms.Compose([
        transforms.Resize((ckpt['size'], ckpt['size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = Subset(
        datasets.ImageFolder(base, transform=transform),
        val_ids
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)

            logits = model(x)
            predictions = logits.argmax(dim=1).cpu().numpy()
            actual = y.numpy()

            for true_label, predicted_label in zip(
                actual, predictions
            ):
                confusion[true_label, predicted_label] += 1

    # Overall validation accuracy.
    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    accuracy = correct / total

    print()
    print('VALIDATION ERROR ANALYSIS')
    print('=' * 55)
    print(f'Model: finetune_v4')
    print(f'Image size: {ckpt["size"]}')
    print(f'Overall accuracy: {accuracy * 100:.2f}%')
    print(f'Correct: {correct}/{total}')
    print()

    # Per-class accuracy.
    results = []

    for i, name in enumerate(classes):
        class_total = int(confusion[i].sum())
        class_correct = int(confusion[i, i])

        class_accuracy = (
            class_correct / class_total
            if class_total else 0.0
        )

        results.append({
            'class': name,
            'correct': class_correct,
            'total': class_total,
            'accuracy': class_accuracy
        })

    results.sort(key=lambda row: row['accuracy'])

    print('PER-CLASS ACCURACY (LOWEST FIRST)')
    print('-' * 55)

    for row in results:
        print(
            f'{row["class"]:<25} '
            f'{row["correct"]:>3}/{row["total"]:<3} '
            f'{row["accuracy"] * 100:>6.2f}%'
        )

    # Save per-class accuracy.
    output_csv = args.output / 'v4_per_class_accuracy.csv'

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['class', 'correct', 'total', 'accuracy']
        )
        writer.writeheader()
        writer.writerows(results)

    # Save confusion matrix.
    matrix_csv = args.output / 'v4_confusion_matrix.csv'

    with open(matrix_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual/predicted'] + classes)

        for i, name in enumerate(classes):
            writer.writerow(
                [name] + confusion[i].tolist()
            )

    # Identify the most frequent class confusions.
    errors = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                errors.append((
                    int(confusion[i, j]),
                    classes[i],
                    classes[j]
                ))

    errors.sort(reverse=True)

    print()
    print('TOP 10 CLASS CONFUSIONS')
    print('-' * 55)

    for count, actual, predicted in errors[:10]:
        print(
            f'{actual} -> {predicted}: {count} errors'
        )

    print()
    print('Saved:', output_csv)
    print('Saved:', matrix_csv)


def main():
    p = argparse.ArgumentParser()

    # Training stages
    p.add_argument(
        '--stage',
        choices=[
           'prepare', 'baseline', 'ssl', 'linear',
           'finetune', 'finetune_v2', 'finetune_v3', 'finetune_v4',
           'evaluate', 'predict', 'analyze'
        ],
        required=True
    )

    # Dataset and output paths
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=None)

    # Model selection
    p.add_argument(
        '--model',
        choices=['baseline', 'linear', 'finetune', 'finetune_v2', 'finetune_v3', 'finetune_v4'],
        default='finetune'
    )

    # Training configuration
    p.add_argument('--size', type=int, default=96)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--ssl-epochs', type=int, default=80)

    # Learning rates
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--ssl-lr', type=float, default=0.0003)

    # Experiment 4 hyperparameters
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--encoder-lr-factor', type=float, default=0.1)

    # SSL and validation configuration
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)

    # Parse arguments
    args = p.parse_args()

    args.root = args.root.resolve()
    args.output = (args.output or args.root / 'results').resolve()

    args.output.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print('Device:', args.device, 'PyTorch:', torch.__version__)

    # Verify datasets
    
    prepare(args.root)

    if args.stage == 'prepare':
        return

    # Experiment 7: Validation error analysis
    if args.stage == 'analyze':
        analyze_validation(args)
        return

    # Evaluation and prediction
    if args.stage in ('evaluate', 'predict'):
        if args.stage == 'evaluate':
            evaluate(args)
        else:
            predict(args)
        return


    # Load training and validation datasets
    tr, va, args.classes = loaders(
        args,
        'ssl' if args.stage == 'ssl' else 'supervised'
    )

    # Save experiment configuration
    (args.output / 'config.json').write_text(
        json.dumps(
            {k: str(v) for k, v in vars(args).items()},
            indent=2
        )
    )

    # Self-supervised pretraining
    if args.stage == 'ssl':
        run_ssl(args, tr)
        return

    # Initialize classifier
    model = Classifier(len(args.classes))

    # Load pretrained SSL encoder for downstream experiments
    if args.stage in ('linear', 'finetune', 'finetune_v2', 'finetune_v3', 'finetune_v4'):
        ckpt = torch.load(
            args.output / 'ssl_encoder.pt',
            map_location='cpu',
            weights_only=True
        )

        model.backbone.load_state_dict(ckpt['backbone'])

    # Supervised training
    train_supervised(
        model,
        tr,
        va,
        args,
        args.stage,
        freeze=args.stage == 'linear'
    )


if __name__=='__main__':main()
    