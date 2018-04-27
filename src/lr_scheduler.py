import logging
import warnings

import numpy as np
from torch.optim.optimizer import Optimizer

logger = logging.getLogger('root')


class ReduceLROnPlateau(object):
    """
    Reduce learning rate when the metric plateaus. 
    Args:
    optimizer: instance of torch optimizer
    mode: {min, max}. In min mode the lr will be reduced when the quantity being monitored stops reducing by a margin of epsilon.
    epsilon: threshold for measuring new optimum
    cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
    min_lr: lower bound on the learning rate. 

    Usage:
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
    >>> for epoch in range(10):
    >>>   train(...)
    >>>   val_acc, val_loss = validate(...)
    >>>   scheduler.step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()
        assert isinstance(optimizer, Optimizer), "optimizer not an instance of torch.optim.optimizer"
        if factor >= 1.0: raise ValueError('ReduceLROnPlateau does not support a factor>=1.0')
        self.finished = False
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        if self.mode not in ['min', 'max']: raise RuntimeError('ReduceLROnPlateau mode:%s not valid.' % self.mode)
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.inf
        else:
            self.monitor_op = lambda a, b: np.less(a, b + self.epsilon)
            self.best = -np.inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics.', RuntimeWarning)
        else:
            if self.in_cooldown():
                logger.info(
                    'ReduceLROnPlateau in cooldown. (%s/%s)' % (self.cooldown - self.cooldown_counter, self.cooldown))
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                logger.info('ReduceLROnPlateau wait reset current_best:%s new_best:%s' % (self.best, current))
                self.best = current
                self.wait = 0

            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            logger.info('ReduceLROnPlateau reducing learning rate to %s.' % new_lr)
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                        else:
                            self.finished = True

                    return True
                else:
                    logger.info('ReduceLROnPlateau wait within patience. (%s/%s)' % (self.wait + 1, self.patience))
                self.wait += 1
        return False

    def in_cooldown(self):
        return self.cooldown_counter > 0
