class BaseEvaluator():
    def __init__(self):
        # need properties used to save result of each batch
        pass

    def reset(self):
        # reset the properties at the beginning of each epoch's validation process
        pass

    def add_batch(self):
        # validation set is tested with multiple batches, save each batch result
        pass

    def Pixel_Accuracy(self):
        # get metric from the property used to save each batch's result
        pass

    def Mean_Intersection_over_Union(self):
        # get metric from the property used to save each batch's result
        pass