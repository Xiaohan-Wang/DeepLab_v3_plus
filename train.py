import warnings
warnings.filterwarnings("ignore")

import mlflow
import os
from torch.utils.tensorboard import SummaryWriter
from models.deeplab import build_Deeplab
from config import Config
from utils.loss.segmentation_loss import build_loss
from datasets import build_dataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import build_lr_scheduler
import torch.optim as optim
from utils.metric.segmentation_metric import build_evaluator
from utils.saver import build_saver
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.config = config
        self.writer = None
        self.saver = None

        # model
        self.model = build_Deeplab(config.backbone, config.output_stride, config.num_classes)
        if config.cuda:
            self.model.to('cuda:1')

        # train
        train_set = build_dataset(config.dataset, config.dataset_root, 'train')
        self.train_loader = DataLoader(train_set, batch_size=config.train_batch_size,
                                       num_workers=config.train_num_workers, shuffle=True, pin_memory=True)
        self.criterion = build_loss(mode=config.loss)(ignore_index=config.ignore_index)
        if config.cuda:
            self.criterion.to('cuda:1')
        self.optimizer = optim.SGD([
            {'params': self.model.get_1x_lr_parameters(), 'lr': config.lr},
            {'params': self.model.get_10x_lr_parameters(), 'lr': config.lr * 10}
        ])

        # validate
        val_set = build_dataset(config.dataset, config.dataset_root, 'train')
        self.val_loader = DataLoader(val_set, batch_size=config.val_batch_size,
                                     num_workers=config.val_num_workers, shuffle=False, pin_memory=True)
        self.evaluator = build_evaluator(config.num_classes)
        self.best_pred = 0

        # resuming checkpoint
        self.start_epoch = 0
        if config.resume is not None:
            if not os.path.exists(config.resume):
                raise RuntimeError("=> No checkpoint found at {}".format(config.resume))
            checkpoint = torch.load(config.resume)
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['model'])
            self.best_pred = checkpoint['pred']
        self.lr_scheduler = build_lr_scheduler(mode=config.lr_scheduler)(self.optimizer,
                                                                         max_iters=config.max_epoch * len(
                                                                             self.train_loader),
                                                                         warmup_factor=0.001,
                                                                         warmup_iters=config.warmup_epoch * len(
                                                                             self.train_loader),
                                                                         warmup_method="linear",
                                                                         last_epoch=self.start_epoch * len(
                                                                             self.train_loader) - 1)

    def train(self):
        """
        implement the logic of train:
        -prepare mlflow
        -log experiment parameters
        -loop
            -train_epoch
            -val_epoch
        """
        client = mlflow.tracking.MlflowClient()
        mlflow.set_experiment(self.config.exp_name)
        exp_id = client.get_experiment_by_name(self.config.exp_name).experiment_id
        print("exp_name:{}, exp_id:{}".format(self.config.exp_name, exp_id))
        with mlflow.start_run(run_name=self.config.run_name):
            self.prepare_artifacts_dir(mlflow.get_artifact_uri())
            # log parameters
            for name in dir(self.config):
                if not name.startswith('__'):
                    mlflow.log_param(name, getattr(self.config, name))
            for epoch in range(self.start_epoch, self.config.max_epoch):
                self.train_epoch(epoch)
                self.val_epoch(epoch)

    def train_epoch(self, epoch):
        """
        implement the logic of train epoch:
        -self.model.train()
        -loop over training samples
        -log any metrics you want using the mlflow/tensorboard
        """
        self.model.train()
        total_train_loss = 0
        num_images = 0
        for i, sample in enumerate(tqdm(self.train_loader)):
            iter = epoch * len(self.train_loader) + i
            image, target = sample[0], sample[1]
            if self.config.cuda:
                image, target = image.to('cuda:1'), target.to('cuda:1')
            output = self.model(image)
            loss = self.criterion(output, target)
            total_train_loss += loss.item() * image.shape[0]
            num_images += image.shape[0]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.writer.add_scalar('1x_lr', self.optimizer.param_groups[0]['lr'], iter)
            self.writer.add_scalar('10x_lr', self.optimizer.param_groups[1]['lr'], iter)
            self.writer.add_scalar('iter_train_loss', loss.item(), iter)
        self.writer.add_scalar('epoch_train_loss', total_train_loss / num_images, epoch)
        print("epoch {}, image_num {}, train loss {:.2f}".format(epoch, num_images, total_train_loss / num_images))

    def val_epoch(self, epoch):
        """
        implement the logic of val epoch:
        -self.model.eval()
        -loop over validation samples
        -log any metrics you want using the mlflow
        -save models if better than best_pred
        """
        self.model.eval()
        num_images = 0
        total_val_loss = 0
        self.evaluator.reset()
        for i, sample in enumerate(tqdm(self.val_loader)):
            image, target = sample[0], sample[1]
            if self.config.cuda:
                image, target = image.to('cuda:1'), target.to('cuda:1')
            with torch.no_grad():
                output = self.model(image)
            # loss
            loss = self.criterion(output, target)
            num_images += image.shape[0]
            total_val_loss += loss.item() * image.shape[0]
            # metrics
            pred = torch.argmax(output, dim=1)
            self.evaluator.add_batch(target, pred)
        self.writer.add_scalar('epoch_val_loss', total_val_loss / num_images, epoch)
        Acc = self.evaluator.get_Pixel_Accuracy()
        mAcc = self.evaluator.get_Mean_Pixel_Accuracy()
        mIoU = self.evaluator.get_Mean_Intersection_over_Union()
        fwIoU = self.evaluator.get_Frequency_Weighted_Intersection_over_Union()
        self.log_metric('Acc', Acc, epoch)
        self.log_metric('mAcc', mAcc, epoch)
        self.log_metric('mIoU', mIoU, epoch)
        self.log_metric('fwIoU', fwIoU, epoch)
        print("epoch {}, image_num {}, val loss {:.2f}".format(epoch, num_images, total_val_loss / num_images))
        print("Acc:{:.2f}, mAcc:{:.2f}, mIoU:{:.2f}, fwIoU:{:.2f}".format(Acc, mAcc, mIoU, fwIoU))
        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_state({
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'pred': mIoU
            }, epoch)

    def log_metric(self, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step)

    def prepare_artifacts_dir(self, artifacts_dir):
        if artifacts_dir.startswith("file://"):
            artifacts_dir = artifacts_dir[7:]
        # model saver
        model_dir = os.path.join(artifacts_dir, "models")
        os.mkdir(model_dir)
        self.saver = build_saver(model_dir)
        # tensorboard
        tb_dir = os.path.join(artifacts_dir, 'events')
        os.mkdir(tb_dir)
        self.writer = SummaryWriter(tb_dir)
        print("For MLflow, go http://localhost:5000.")
        print("For Tensorboard, use tensorboard --logdir={} --port=6007, then go http://localhost:6007.".format(tb_dir))
        return artifacts_dir

if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()

