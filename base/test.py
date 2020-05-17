from base_trainer import BaseTrainer
import argparse
import os
import mlflow
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

class Config:
    config_id = 6
    test_prop = True
    temp_dir = '/usr/xtmp/xw176/temp'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)



class Trainer(BaseTrainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.model = Net()

        enable_cuda_flag = True if args.enable_cuda == 'True' else False
        args.cuda = enable_cuda_flag and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        if args.cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        self.model_temp_dir = tempfile.mkdtemp(dir=self.config.temp_dir)
        print("Writing models locally to %s\n" % self.model_temp_dir)
        # Create a SummaryWriter to write TensorBoard events locally
        self.tb_temp_dir = tempfile.mkdtemp(dir=self.config.temp_dir)
        self.writer = SummaryWriter(self.tb_temp_dir)
        print("Writing TensorBoard events locally to %s\n" % self.tb_temp_dir)

    def train_epoch(self, epoch):
        """
        implement the logic of train epoch:
        -loop over training samples
        -log any metrics you want using the mlflow
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))
                step = epoch * len(self.train_loader) + batch_idx
                self.log_metric('train_loss', loss.data.item(), step)


    def val_epoch(self, epoch):
        """
        implement the logic of val epoch:
        -loop over validation samples
        -log any metrics you want using the mlflow
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').data.item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100.0 * correct / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), test_accuracy))
        step = (epoch + 1) * len(self.train_loader)
        self.log_metric('test_loss', test_loss, step)
        self.log_metric('test_accuracy', test_accuracy, step)

    def train(self):
        """
        implement the logic of train:
        -log experiment parameters
        -train_epoch
        -val_epoch
        """
        with mlflow.start_run():
            print("tracking URI: ", mlflow.tracking.get_tracking_uri())
            # Log our parameters into mlflow
            for key, value in vars(self.args).items():
                mlflow.log_param(key, value)
            for name in dir(self.config):
                if not name.startswith('__'):
                    mlflow.log_param(name, getattr(self.config, name))


            # Perform the training
            for epoch in range(1, args.epochs + 1):
                self.train_epoch(epoch)
                self.val_epoch(epoch)
                if epoch % 2 == 0 and epoch!=0:
                    torch.save(self.model.state_dict(), self.model_temp_dir+'/epoch_'+str(epoch)+'.pth')

            print("Uploading models as a run artifact...")
            mlflow.log_artifacts(self.model_temp_dir, artifact_path="models")

            # Upload the TensorBoard event logs as a run artifact
            print("Uploading TensorBoard events as a run artifact...")
            mlflow.log_artifacts(self.tb_temp_dir, artifact_path="events")
            print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
                  os.path.join(mlflow.get_artifact_uri(), "events"))


if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='True',
                        help='enables or disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    config = Config()

    trainer = Trainer(config, args)
    trainer.train()



