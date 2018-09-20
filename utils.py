import tensorflow as tf
import time 

class LoggerHook(tf.train.SessionRunHook):
  """Logs loss and runtime."""
  def __init__(self, log_frequency, batch_size):
    self.log_frequency = log_frequency
    self.batch_size = batch_size
    super(LoggerHook, self).__init__()

  def begin(self):
    self._start_time = time.time()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[
      tf.train.get_global_step(),
      'loss:0'])

  def after_run(self, run_context, run_values):
    step, loss = run_values.results
    step += 1
    if step % self.log_frequency == 0:
      current_time = time.time()
      duration = current_time - self._start_time

      examples_per_sec = step * self.batch_size / duration
      sec_per_batch = duration / step

      format_str = "Step {}: {:0.1f} examples/sec; {:0.4f} sec/batch\n"
      print(format_str.format(step, examples_per_sec, sec_per_batch))
