from base.base_trainer import BaseTrainer
from torch import optim
import tempfile
from torch.utils.tensorboard import SummaryWriter
from models.deeplab import DeepLab


class Trainer(BaseTrainer):
    def __init__(self, config, args):
        super(Trainer, self).__init__(config, args)
        self.model = DeepLab(config.backbone, config.output_stride, config.num_classes)
        self.train_loader = None
        self.val_loader = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.evaluator = None




        self.tb_temp_dir = tempfile.mkdtemp(dir=self.config.temp)
        self.writer = SummaryWriter(self.tb_temp_dir)
        self.model_temp_dir = tempfile.mkdtemp(dir=self.config.temp)

    def train_epoch(self, epoch):
        # TODO: train_epoch
        """
        implement the logic of train epoch:
        -self.model.train()
        -loop over training samples
        -log any metrics you want using the mlflow
        """
        raise NotImplementedError

    def val_epoch(self, epoch):
        # TODO: val_epoch
        """
        implement the logic of val epoch:
        -self.model.eval()
        -loop over validation samples
        -log any metrics you want using the mlflow
        """
        raise NotImplementedError

    def train(self):
        # TODO: train
        """
        implement the logic of train:
        -print("tracking URI: ", mlflow.tracking.get_tracking_uri())
        -log experiment parameters
        -loop
            -train_epoch and
            -val_epoch
            -save models
        -log 1) tensorboard output dir and 2)model dir as artifacts
        """
        raise NotImplementedError


