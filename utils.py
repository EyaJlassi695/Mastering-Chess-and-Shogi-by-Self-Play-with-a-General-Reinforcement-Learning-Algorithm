class AverageMeter(object):
    """Tracks and computes the average of values over time."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        """Updates the meter with a new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    """Allows dot notation access to dictionary attributes."""

    def __getattr__(self, name):
        return self[name]
