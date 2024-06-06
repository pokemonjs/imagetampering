from paddle.optimizer.lr import LinearWarmup

from paddle.optimizer.lr import CosineAnnealingDecay


class Cosine(CosineAnnealingDecay):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        super(Cosine, self).__init__(
            learning_rate=lr,
            T_max=step_each_epoch * epochs, verbose=True)

        self.update_specified = False

class CosineWarmup(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
            epochs, warmup_epoch)
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)

        super(CosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        self.update_specified = False