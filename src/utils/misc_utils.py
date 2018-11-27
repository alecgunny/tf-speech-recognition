import tensorflow as tf


class ThroughputHook(tf.train.StepCounterHook):
  def __init__(self, batch_size, **kwargs):
    self.batch_size = batch_size
    super(ThroughputHook, self).__init__(**kwargs)

  def begin(self):
    super(ThroughputHook, self).begin()
    self._summary_tag = 'throughput'

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    super(ThroughputHook, self)._log_and_record(
        elapsed_steps*self.batch_size,
        elapsed_time,
        global_step)


def get_max_steps(num_examples, batch_size, num_epochs):
  total_examples = num_examples * num_epochs
  return (total_examples - 1) // batch_size + 1

