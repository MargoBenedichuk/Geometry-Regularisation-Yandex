from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import hyptorch.nn as hypnn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.dim)
        self.tp = hypnn.ToPoincare(
            c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=args.dim, n_classes=10, c=args.c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tp(x)
        return (x, F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1))


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        embedding, output = model(data)
        loss = criterion(output, target) + hypnn.geometricReg(embedding, _lambda=args.lambda1, c=args.c)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        losses.append(loss.item())
    return losses


def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += criterion(
                output, target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return test_loss, correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
    )
    parser.add_argument(
        "--lambda1", type=float, default=1.0, help="Lambda parameter for regularization"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
    )
    parser.add_argument(
        "--train_x",
        action="store_true",
        default=False,
        help="train the exponential map origin",
    )
    parser.add_argument(
        "--train_c",
        action="store_true",
        default=False,
        help="train the Poincare ball curvature",
    )
    parser.add_argument(
        "--savefig", default="plot.png"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = Net(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss = []
    test_accuracy = []
    test_loss = []
    for epoch in range(1, args.epochs + 1):
        ls = train(args, model, nn.NLLLoss(), device, train_loader, optimizer, epoch)
        v, a = test(args, model, nn.NLLLoss(reduction='sum'), device, test_loader)
        train_loss.append(sum(ls) / len(ls))
        test_accuracy.append(a)
        test_loss.append(v)

    # --- 1. Настройка стиля ---
    # Используем стиль из seaborn для приятного внешнего вида
    sns.set_style("whitegrid")
    # Устанавливаем хороший шрифт
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # --- 2. Генерация реалистичных данных ---
    # В реальной жизни вы будете использовать свои данные из history.history (Keras) или логов
    epochs = np.arange(1, args.epochs+1)

# --- 3. Создание фигуры и осей ---
    # Создаем фигуру с двумя графиками (один под другим)
    # figsize задает размер в дюймах для хорошего разрешения
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- 4. Построение графиков ---

    # График 1: Потери (Loss)
    axes[0].plot(epochs, train_loss, label='Ошибка на обучении (Train Loss)', color='#4E79A7', marker='o', markersize=4, linestyle='-')
    axes[0].plot(epochs, test_loss, label='Ошибка на тесте (Test Loss)', color='#F28E2B', marker='x', markersize=4, linestyle='--')

    # Настройка первого графика
    axes[0].set_title('Динамика функции потерь', fontsize=16, pad=20)
    axes[0].set_xlabel('Эпохи', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=12)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # График 2: Точность (Accuracy)
    axes[1].plot(epochs, test_accuracy, label='Точность на тесте (Test Accuracy)', color='#59A14F', marker='s', markersize=4, linestyle='-')

    # Настройка второго графика
    axes[1].set_title('Динамика точности', fontsize=16, pad=20)
    axes[1].set_xlabel('Эпохи', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=12)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    # Устанавливаем пределы по оси Y для точности от 0 до 1
    axes[1].set_ylim(0, 1.05)


    # --- 5. Финальные штрихи ---
    # Автоматически подгоняем элементы, чтобы они не пересекались
    # rect=[left, bottom, right, top] - оставляет место для общего заголовка
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Показать график
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
