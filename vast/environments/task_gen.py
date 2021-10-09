import numpy

class Taskgen(object):

    def __init__(self, nr_buckets, bucket_size):
        self.nr_buckets = nr_buckets
        self.bucket_size = bucket_size

    def generate(self):
        raise NotImplementedError

class TaskgenRange(Taskgen):

    def __init__(self, nr_buckets, bucket_size, process_type_low, process_type_high):
        super(TaskgenRange, self).__init__(nr_buckets, bucket_size)
        self.nr_buckets = nr_buckets
        self.bucket_size = bucket_size
        self.process_type_low = process_type_low
        self.process_type_high = process_type_high
        self.process_range = numpy.arange(process_type_low, process_type_high+1)

    def generate(self):
        tasks = []
        for _ in range(self.nr_buckets):
            bucket = numpy.random.choice(self.process_range, size=self.bucket_size, replace=True).tolist()
            tasks.append(bucket)
        return tasks

    def __repr__(self):
        return "TaskgenRange: {}".format(vars(self))

class TaskgenRangeFixedLasttask(Taskgen):

    def __init__(self, nr_buckets, bucket_size, process_type_low, process_type_high, process_type_last):
        super(TaskgenRangeFixedLasttask, self).__init__(nr_buckets, bucket_size)
        self.nr_buckets = nr_buckets
        self.bucket_size = bucket_size
        self.process_type_low = process_type_low
        self.process_type_high = process_type_high
        self.process_range = numpy.arange(process_type_low, process_type_high+1)
        self.process_type_last = process_type_last

    def generate(self):
        tasks = []
        for _ in range(self.nr_buckets - 1):
            bucket = numpy.random.choice(self.process_range, size=self.bucket_size, replace=True).tolist()
            tasks.append(bucket)
        tasks.append([self.process_type_last])
        return tasks

    def __repr__(self):
        return "TaskgenRange: {}".format(vars(self))

class TaskgenRangeExcept(Taskgen):

    def __init__(self, nr_buckets, bucket_size, process_type_low, process_type_high, except_processes):
        super(TaskgenRangeExcept, self).__init__(nr_buckets, bucket_size)
        self.nr_buckets = nr_buckets
        self.bucket_size = bucket_size
        self.process_type_low = process_type_low
        self.process_type_high = process_type_high
        self.except_processes = except_processes
        self.process_range = numpy.setdiff1d(numpy.arange(process_type_low, process_type_high+1), except_processes)

    def generate(self):
        tasks = []
        for _ in range(self.nr_buckets):
            bucket = numpy.random.choice(self.process_range, size=self.bucket_size, replace=True).tolist()
            tasks.append(bucket)
        return tasks

    def __repr__(self):
        return "TaskgenRangeExcept: {}".format(vars(self))