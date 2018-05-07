# A pseudo Horovod class in case Horovod cannot be imported.

class hvd():

    def __init__(self):
        size = 1
        local_rank = 0
        rank = 0

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
