class FeatureBaseClass(object):

    __feature_name__ = "undefined"

    def __init__(self):
        self.feature = None
        self.trace = None

    def compute(self):
        msg = "compute is not implemented yet"
        raise NotImplementedError(msg)
