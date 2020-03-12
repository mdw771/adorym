# A pseudo Horovod class in case Horovod cannot be imported.

class Hvd(object):

    def __init__(self):
        pass

    def init(self):
        pass

    def size(self):
        return 1

    def rank(self):
        return 0

    def local_rank(self):
        return 0

    def broadcast_global_variables(self, a):
        pass

    def DistributedOptimizer(self, op, name=None):
        return op


class Mpi(object):

    def __init__(self):
        pass

    def Barrier(self):
        pass
