import mlflow

class BaseTrainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.writer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.tb_temp_dir = None
        self.model_temp_dir = None

    def train_epoch(self, epoch):
        """
        implement the logic of train epoch:
        -self.model.train()
        -loop over training samples
        -log any metrics you want using the mlflow
        """
        raise NotImplementedError

    def val_epoch(self, epoch):
        """
        implement the logic of val epoch:
        -self.model.eval()
        -loop over validation samples
        -log any metrics you want using the mlflow
        """
        raise NotImplementedError

    def train(self):
        """
        implement the logic of train:
        -log experiment parameters
        -loop
            -train_epoch and
            -val_epoch
            -save models
        -log 1) tensorboard output dir and 2)model dir as artifacts
        """
        raise NotImplementedError

    def log_metric(self, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step)
