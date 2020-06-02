from sklearn.metrics import confusion_matrix
import numpy as np


class SegmentationEvaluator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros(self.num_classes, self.num_classes)

    def add_batch(self, target, pred):
        mask = (target >= 0) & (target < self.num_classes)
        target = target[mask]
        pred = pred[mask]
        batch_confusion_matrix = confusion_matrix(target, pred)
        self.confusion_matrix += batch_confusion_matrix

    def get_Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def get_Mean_Pixel_Accuracy(self):
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # 0/0 = nan, use nanmean to ignore nan items
        mAcc = np.nanmean(class_acc)
        return mAcc

    def get_Mean_Intersection_over_Union(self):
        sum_gt = self.confusion_matrix.sum(axis=1)
        sum_pred = self.confusion_matrix.sum(axis=0)
        tp = np.diag(self.confusion_matrix)
        class_IoU = tp / (sum_gt + sum_pred - tp)
        mIoU = np.nanmean(class_IoU)
        return mIoU

    def get_Frequency_Weighted_Intersection_over_Union(self):
        sum_gt = self.confusion_matrix.sum(axis=1)
        sum_pred = self.confusion_matrix.sum(axis=0)
        tp = np.diag(self.confusion_matrix)
        class_IoU = tp / (sum_gt + sum_pred - tp)
        weighted_IoU = ((sum_gt / sum_gt.sum()) * class_IoU).sum()
        return weighted_IoU


def build_evaluator(num_classes):
    return SegmentationEvaluator(num_classes)


if __name__ == '__main__':
    target = np.array([[[3,0],[2,1]],[[1,2],[2,3]]])
    pred = np.array([[[3,3],[1,0]],[[1,2],[2,2]]])
    evaluator = SegmentationEvaluator(num_classes=4)
    evaluator.add_batch(target, pred)
    acc = evaluator.get_Pixel_Accuracy()
    macc = evaluator.get_Mean_Pixel_Accuracy()
    mIoU = evaluator.get_Mean_Intersection_over_Union()
    wIoU = evaluator.get_Frequency_Weighted_Intersection_over_Union()
    print("confusion matrix\n", evaluator.confusion_matrix)
    print("acc:{}\nmacc:{}\nmIoU:{}\nwIoU:{}".format(acc, macc, mIoU, wIoU))


# correct answer
# confusion matrix
#  [[0. 0. 0. 1.]
#  [1. 1. 0. 0.]
#  [0. 1. 2. 0.]
#  [0. 0. 1. 1.]]
# acc:0.5
# macc:0.41666666666666663
# mIoU:0.29166666666666663
# wIoU:0.35416666666666663

