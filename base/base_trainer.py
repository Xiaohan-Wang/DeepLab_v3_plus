import mlflow
import os
from torch.utils.tensorboard import SummaryWriter
from utils.saver import build_saver
import torch

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.writer = None
        self.saver = None

        # model
        self.model = None

        # train
        self.train_loader = None
        self.criterion = None
        self.optimizer = None

        # validate
        self.val_loader = None
        self.evaluator = None
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
        self.lr_scheduler = None # put it here because it need last_epoch=start_epoch-1 to compute current status

    def train(self):
        """
        implement the logic of train:
        -prepare mlflow
        -log experiment parameters
        -loop
            -train_epoch
            -val_epoch
        """
        raise NotImplementedError

    def train_epoch(self, epoch):
        """
        implement the logic of train epoch:
        -self.model.train()
        -loop over training samples
        -log any metrics you want using the mlflow/tensorboard
        """
        raise NotImplementedError

    def val_epoch(self, epoch):
        """
        implement the logic of val epoch:
        -self.model.eval()
        -loop over validation samples
        -log any metrics you want using the mlflow
        -save models if better than best_pred
        """
        raise NotImplementedError

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
