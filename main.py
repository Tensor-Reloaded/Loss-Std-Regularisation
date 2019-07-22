import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from models import *
from misc import progress_bar
from learn_utils import begin_chart, begin_per_epoch_chart, add_chart_point, reset_seed


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--std_pen', default=2.0, type=float, help='learning rate')
    parser.add_argument('--multi_loss_lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--model', default="VGG('VGG19')", type=str, help='what model to use')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=512, type=int, help='testing batch size')
    parser.add_argument('--num_workers_train', default=4, type=int, help='number of workers for loading train data')
    parser.add_argument('--num_workers_test', default=2, type=int, help='number of workers for loading test data')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--std_loss', '-std', action='store_true', help='add std loss')
    parser.add_argument('--per_class_std', '-pc_std', action='store_true', help='compute std per class')
    parser.add_argument('--train_batch_plot_freq', default=20, type=int, help='freq to plot batch statistics')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.train_batch_idx = 0
        self.test_batch_idx = 0

    def get_train_batch_idx(self):
        self.train_batch_idx += 1
        return self.train_batch_idx - 1

    def get_test_batch_idx(self):
        self.test_batch_idx += 1
        return self.test_batch_idx - 1

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../storage', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='../storage', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model).to(self.device)
        self.multi_loss = MultiLossModel(2).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.multi_loss_optimizer = optim.SGD(self.multi_loss.parameters(), lr = self.args.multi_loss_lr, momentum=0.2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90, 120, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        total_std = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            self.multi_loss_optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss_mean = loss.mean()
            loss_std = loss.std()
            total_std += loss_std.item()

            if batch_num % self.args.train_batch_plot_freq == 0:
                plot_idx = self.get_train_batch_idx()
                add_chart_point("TrainPerBatchStd", plot_idx, loss_std.item())
                add_chart_point("TrainPerBatchMean", plot_idx, loss_mean.item())
                add_chart_point("MeanWeight", plot_idx, self.multi_loss.weights[0].item())
                add_chart_point("StdWeight", plot_idx, self.multi_loss.weights[1].item())

            if self.args.std_loss:
                if self.args.per_class_std:
                    class_count = 0
                    current_std = 0.0
                    for i in range(len(CLASSES)):
                        if loss[target == i].size(0) > 2:
                            current_std = current_std + loss[target == i].std()
                            class_count += 1
                    loss = loss_mean + current_std / class_count
                else:
                    loss = loss_mean + self.args.std_pen * loss_std
                    # loss = self.multi_loss(torch.cat([loss_mean.unsqueeze(0), loss_std.unsqueeze(0)]))

            else:
                loss = loss_mean
            loss.backward()
            self.optimizer.step()
            self.multi_loss_optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            # progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss / len(self.train_loader), train_correct / total, total_std / len(self.train_loader)

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        total_std = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                plot_idx = self.get_test_batch_idx()
                loss_mean = loss.mean()
                loss_std = loss.std()

                total_std += loss_std.item()

                add_chart_point("TestPerBatchStd", plot_idx, loss_std.item())
                add_chart_point("TestPerBatchMean", plot_idx, loss_mean.item())

                loss = loss_mean
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                # progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss / len(self.test_loader), test_correct / total , total_std / len(self.test_loader)

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()

        begin_per_epoch_chart("TrainAcc")
        begin_per_epoch_chart("TestAcc")
        begin_per_epoch_chart("TrainLoss")
        begin_per_epoch_chart("TestLoss")

        begin_chart("TrainPerBatchMean", "BatchIdx")
        begin_chart("TestPerBatchMean", "BatchIdx")

        begin_chart("TrainPerBatchStd", "BatchIdx")
        begin_chart("TestPerBatchStd", "BatchIdx")

        begin_per_epoch_chart("TrainStd")
        begin_per_epoch_chart("TestStd")

        begin_chart("StdWeight", "BatchIdx")
        begin_chart("MeanWeight", "BatchIdx")

        reset_seed(0)
        accuracy = 0
        for epoch in range(1, self.args.epoch + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            add_chart_point("TrainAcc", epoch, train_result[1])
            add_chart_point("TrainLoss", epoch, train_result[0])
            add_chart_point("TrainStd", epoch, train_result[2])
            print(train_result)
            test_result = self.test()

            add_chart_point("TestAcc", epoch, test_result[1])
            add_chart_point("TestLoss", epoch, test_result[0])
            add_chart_point("TestStd", epoch, test_result[2])

            accuracy = max(accuracy, test_result[1])
            if epoch == self.args.epoch:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
