"""
LegendreTron on MNIST datasets
"""
import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import lib.data
import lib.models

torch.set_default_dtype(torch.float64)
_mnists = ["mnist", "fmnist", "kmnist"]


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx, :]
        image = self.images[idx, :]
        return image, label


# parsing arguments
parser = argparse.ArgumentParser("LegendreTron demo on MNIST datasets")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="mnist", choices=_mnists)
parser.add_argument("--dimh", type=int, default=4)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--blocks", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--gamma_every", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size_train", type=int, default=128)
parser.add_argument("--batch_size_test", type=int, default=64)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--binary", action="store_true")
parser.add_argument("--pos", nargs="+", type=int, default=[0])
parser.add_argument("--neg", nargs="*", type=int)
parser.add_argument("--ntrain", nargs="?", type=int)
parser.add_argument("--ntest", nargs="?", type=int)

args = parser.parse_args()

# read args
seed = args.seed
dataset = args.dataset
dimh = args.dimh
depth = args.depth
blocks = args.blocks
lr = args.lr
gamma = args.gamma
gamma_every = args.gamma_every
weight_decay = args.weight_decay
num_epochs = args.num_epochs
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
print_every = args.print_every
binary = args.binary
pos = args.pos  # positive class label(s)
neg = args.neg  # negative class label(s) (None means complement of pos)
ntrain = args.ntrain  # number of training points
ntest = args.ntest  # number of testing points

# setting seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# data setup
data_path = os.path.expanduser("data")
lib.data._path = data_path

ntrain = None  # number of training points
ntest = None  # number of testing points
(
    Xtrain,
    Ytrain,
    Ytrainmulti,
    Xtest,
    Ytest,
    Ytestmulti,
) = lib.data.digit_data(pos, neg, ntrain, ntest, dataset, seed)

if not binary:
    Ytrain, Ytest = Ytrainmulti, Ytestmulti
d = Xtrain.shape[1]
K = np.max(Ytrain)

data_train = ImageDataset(Xtrain, Ytrain)
data_test = ImageDataset(Xtest, Ytest)

# model setup
inner_model = lib.models.MultinomialLogisticRegression(
    input_dim=d, output_dim=K
)
link_model = lib.models.LegendreLink(
    n_blocks=blocks, K=K, dim_hidden=dimh, depth=depth
)

# training setup
categorical_loss = torch.nn.NLLLoss(reduction="mean")
train_kwargs = {"batch_size": batch_size_train, "shuffle": True}
test_kwargs = {"batch_size": batch_size_test, "shuffle": True}
cuda = torch.cuda.is_available()

if cuda:
    inner_model = inner_model.cuda()
    link_model = link_model.cuda()
    cuda_kwargs = {"num_workers": 4, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_loader = DataLoader(data_train, **train_kwargs)
test_loader = DataLoader(data_test, **test_kwargs)

model_parameters = list(inner_model.parameters()) + list(
    link_model.parameters()
)
opt = torch.optim.Adam(model_parameters, lr=lr)
sch = torch.optim.lr_scheduler.StepLR(
    opt, step_size=gamma_every * len(train_loader), gamma=gamma
)

# training loop
t0 = time.time()

for e in range(1, num_epochs + 1):
    # optimization step
    inner_model.train()
    link_model.train()
    for _, (X_train, Y_train) in enumerate(
        tqdm(train_loader, desc=f"epoch {e} minibatches")
    ):
        if cuda:
            X_train, Y_train = X_train.cuda(), Y_train.cuda()
        X_train, Y_train = X_train.type(torch.double), Y_train.flatten().type(
            torch.long
        )
        opt.zero_grad()
        logp_pred = link_model(inner_model(X_train), context=None)
        loss_train = categorical_loss(logp_pred, Y_train)

        # L2 regularisation on all parameters
        regulariser = torch.sum(torch.zeros(1, dtype=torch.float))
        for p in model_parameters:
            regulariser = regulariser + torch.sum(p**2)

        if cuda:
            loss_train = loss_train.cuda()
            regulariser = regulariser.cuda()

        regularised_loss = loss_train + 0.5 * weight_decay * regulariser

        regularised_loss.backward()
        opt.step()
        sch.step()

    if e % print_every == 0:
        inner_model.eval()
        link_model.eval()
        # print step
        y_true_train, logp_pred_train, y_pred_train = [], [], []
        y_true_test, logp_pred_test, y_pred_test = [], [], []
        with torch.no_grad():
            for X_train, Y_train in train_loader:  # training data
                if cuda:
                    X_train, Y_train = X_train.cuda(), Y_train.flatten().cuda()
                X_train, Y_train = X_train.type(
                    torch.double
                ), Y_train.flatten().type(torch.long)
                logp_pred = link_model(inner_model(X_train), context=None)
                Y_pred = logp_pred.argmax(dim=-1)

                y_true_train.extend(Y_train.flatten().tolist())
                logp_pred_train.extend(logp_pred.tolist())
                y_pred_train.extend(Y_pred.flatten().tolist())

            y_true_train, logp_pred_train, y_pred_train = (
                torch.as_tensor(y_true_train),
                torch.as_tensor(logp_pred_train),
                torch.as_tensor(y_pred_train),
            )
            loss_train = categorical_loss(logp_pred_train, y_true_train)
            prop_correct_train = (
                torch.sum(y_pred_train == y_true_train) / y_pred_train.shape[0]
            )

            for X_test, Y_test in test_loader:  # testing data
                if cuda:
                    X_test, Y_test = X_test.cuda(), Y_test.flatten().cuda()
                X_test, Y_test = X_test.type(
                    torch.double
                ), Y_test.flatten().type(torch.long)
                logp_pred = link_model(inner_model(X_test), context=None)
                Y_pred = logp_pred.argmax(dim=-1)

                y_true_test.extend(Y_test.flatten().tolist())
                logp_pred_test.extend(logp_pred.tolist())
                y_pred_test.extend(Y_pred.flatten().tolist())

            y_true_test, logp_pred_test, y_pred_test = (
                torch.as_tensor(y_true_test),
                torch.as_tensor(logp_pred_test),
                torch.as_tensor(y_pred_test),
            )
            loss_test = categorical_loss(logp_pred_test, y_true_test)
            prop_correct_test = (
                torch.sum(y_pred_test == y_true_test) / y_pred_test.shape[0]
            )

        if binary:
            y_true_train, p_pred_train = (
                y_true_train.numpy(),
                torch.exp(logp_pred_train).numpy(),
            )
            y_true_test, p_pred_test = (
                y_true_test.numpy(),
                torch.exp(logp_pred_test).numpy(),
            )
            auc_train = roc_auc_score(y_true_train, p_pred_train[:, 1])
            auc_test = roc_auc_score(y_true_test, p_pred_test[:, 1])
            print(
                f"epoch {e} training: auc {100*auc_train}"
                f", pct correct {prop_correct_train:2.2f}"
            )
            print(
                f"epoch {e} test: auc {100*auc_test}"
                f", pct correct {prop_correct_test:2.2f}"
            )
        else:
            print(
                f"epoch {e} training: mean log-likelihood {loss_train.item()}"
                f", pct correct {100 * prop_correct_train:2.2f}"
            )
            print(
                f"epoch {e} testing: mean log-likelihood {loss_test.item()}"
                f", pct correct {100 * prop_correct_test:2.2f}"
            )

elapsed_time = time.time() - t0
print(f"elapsed_time: {elapsed_time}")