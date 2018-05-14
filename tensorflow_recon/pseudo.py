# A pseudo Horovod class in case Horovod cannot be imported.

class hvd(object):

    def __init__(self):
        self.size = 1
        self.local_rank = 0
        self.rank = 0

    def init(self):
        pass

    def size(self):
        return 1

    def rank(self):
        return 0

    def local_rank(self):
        return 0

    def broadcast_global_variables(a):
        pass

    def DistributedOptimizer(self, op):
        return op
