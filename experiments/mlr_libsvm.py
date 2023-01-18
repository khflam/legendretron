"""
run Multinomial Logistic Regression on LIBSVM datasets
"""
import argparse
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.data
import lib.models
from lib.functional import log_softmax_plus

torch.set_default_dtype(torch.float64)
_libsvms = [
    "aloi",
    "dna",
    "glass",
    "iris",
    "letter",
    "news20",
    "satimage",
    "sector",
    "segment",
    "Sensorless",
    "svmguide2",
    "usps",
    "vehicle",
    "vowel",
    "wine",
]


def get_args():
    parser = argparse.ArgumentParser(
        "Multinomial Logistic Regression on LIBSVM datasets"
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="segment", choices=_libsvms
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gamma_every", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=240)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_test", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--flip_labels", action="store_true")
    parser.add_argument("--eta", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run(args):
    # setting seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # data setup
    data_path = os.path.expanduser("data")
    X, Y = lib.data.load_libsvm(data_path, args.dataset)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed
    )
    d = Xtrain.shape[1]
    K = np.max(Y)

    # model setup
    inner_model = lib.models.MultinomialLogisticRegression(
        input_dim=d, output_dim=K
    )

    # training setup
    categorical_loss = torch.nn.NLLLoss(reduction="mean")
    train_kwargs = {"batch_size": args.batch_size_train, "shuffle": True}
    test_kwargs = {"batch_size": args.batch_size_test, "shuffle": True}
    cuda = torch.cuda.is_available()

    if cuda:
        inner_model = inner_model.cuda()
        cuda_kwargs = {"num_workers": 4, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.flip_labels:
        Ytrain = lib.data.noisify(
            Ytrain, K=K, eta=args.eta, random_state=args.seed
        )

    data_train = lib.data.LibSVMDataset(Xtrain, Ytrain)
    data_test = lib.data.LibSVMDataset(Xtest, Ytest)
    train_loader = DataLoader(data_train, **train_kwargs)
    test_loader = DataLoader(data_test, **test_kwargs)

    model_parameters = list(inner_model.parameters())
    opt = torch.optim.Adam(model_parameters, lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.gamma_every * len(train_loader), gamma=args.gamma
    )

    # training loop
    t0 = time.time()

    for e in range(1, args.num_epochs + 1):
        # optimization step
        inner_model.train()
        for _, (X_train, Y_train) in enumerate(
            tqdm(train_loader, desc=f"epoch {e} minibatches")
        ):
            if cuda:
                X_train, Y_train = X_train.cuda(), Y_train.cuda()
            X_train, Y_train = X_train.type(
                torch.double
            ), Y_train.flatten().type(torch.long)
            opt.zero_grad()
            logp_pred = log_softmax_plus(inner_model(X_train))
            loss_train = categorical_loss(logp_pred, Y_train)

            # L2 regularisation on all parameters
            regulariser = torch.sum(torch.zeros(1, dtype=torch.float))
            for p in model_parameters:
                regulariser = regulariser + torch.sum(p**2)

            if cuda:
                loss_train = loss_train.cuda()
                regulariser = regulariser.cuda()

            regularised_loss = (
                loss_train + 0.5 * args.weight_decay * regulariser
            )

            regularised_loss.backward()
            opt.step()
            sch.step()

        if e % args.print_every == 0:
            inner_model.eval()
            # print step
            y_true_train, logp_pred_train, y_pred_train = [], [], []
            y_true_test, logp_pred_test, y_pred_test = [], [], []
            with torch.no_grad():
                for X_train, Y_train in train_loader:  # training data
                    if cuda:
                        X_train, Y_train = (
                            X_train.cuda(),
                            Y_train.flatten().cuda(),
                        )
                    X_train, Y_train = X_train.type(
                        torch.double
                    ), Y_train.flatten().type(torch.long)
                    logp_pred = log_softmax_plus(inner_model(X_train))
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
                    torch.sum(y_pred_train == y_true_train)
                    / y_pred_train.shape[0]
                )

                for X_test, Y_test in test_loader:  # testing data
                    if cuda:
                        X_test, Y_test = X_test.cuda(), Y_test.flatten().cuda()
                    X_test, Y_test = X_test.type(
                        torch.double
                    ), Y_test.flatten().type(torch.long)
                    logp_pred = log_softmax_plus(inner_model(X_test))
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
                    torch.sum(y_pred_test == y_true_test)
                    / y_pred_test.shape[0]
                )

            print(
                f"epoch {e} training: mean llh {loss_train.item()}"
                f", pct correct {100 * prop_correct_train:2.2f}"
            )
            print(
                f"epoch {e} testing: mean llh {loss_test.item()}"
                f", pct correct {100 * prop_correct_test:2.2f}"
            )

    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")


if __name__ == "__main__":
    run(get_args())
