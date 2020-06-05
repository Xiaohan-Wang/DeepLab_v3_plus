import torch
import os

class Saver:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save_state(self, state_dict, epoch):
        # save the best performing model
        torch.save(state_dict, os.path.join(self.model_dir, "epoch_{}.pth".format(epoch)))
        # log best predictions
        best_pred = state_dict['pred']
        with open(os.path.join(self.model_dir, 'best_pred.txt'), 'a+') as f:
            f.write("epoch {}: best pred {:.2f}".format(epoch, best_pred))


def build_saver(model_dir):
    return Saver(model_dir)
