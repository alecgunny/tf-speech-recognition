import tensorflow as tf
from utils import model_utils, data_utils, misc_utils, parse_utils
tf.logging.set_verbosity(0)


def main():
  with open(FLAGS.labels, 'r') as f:
    labels = f.read().split(",")
    labels = labels[:20]

  # build and compile a keras model
  model = model_utils.deepspeech_model(
    FLAGS.input_shape+(1,),
    FLAGS.num_frames,
    FLAGS.frame_step,
    FLAGS.hidden_sizes,
    len(labels)+1,
    FLAGS.dropout)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  print(model.summary())

  # convert to an estimator with given configuration and distribution strategy
  strategy = tf.contrib.distribute.DistributeConfig(
    train_distribute=tf.contrib.distribute.MirroredStrategy(
      num_gpus=FLAGS.num_gpus,
      prefetch_on_device=True))
  config = tf.estimator.RunConfig(
    save_summary_steps=FLAGS.log_steps,
    save_checkpoints_secs=FLAGS.eval_throttle_secs,
    log_step_count_steps=FLAGS.log_steps,
    model_dir=FLAGS.model_dir,
    experimental_distribute=strategy)
  custom_objects = {
    'DeepSpeechCell': model_utils.DeepSpeechCell,
    'ImageToDeepSpeech': model_utils.ImageToDeepSpeech}
  estimator = tf.keras.estimator.model_to_estimator(
    model,
    custom_objects=custom_objects,
    config=config)

  # build data input functions
  mean, var =  data_utils.get_stats(FLAGS.stats, FLAGS.input_shape)
  train_input_fn = data_utils.get_input_fn(
    FLAGS.train_data,
    labels,
    FLAGS.input_shape,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs,
    mean=mean,
    var=var)
  eval_input_fn = data_utils.get_input_fn(
    FLAGS.valid_data,
    labels,
    FLAGS.input_shape,
    batch_size=FLAGS.batch_size*8, # can do bigger batches on inference
    num_epochs=1,
    mean=mean,
    var=var)

  # couple quick utilities
  hooks = [misc_utils.ThroughputHook(
    FLAGS.batch_size*FLAGS.num_gpus,
    every_n_steps=FLAGS.log_steps,
    output_dir=estimator.model_dir)]
  max_steps = misc_utils.get_max_steps(
    num_examples=FLAGS.num_train_examples,
    batch_size=FLAGS.batch_size*FLAGS.num_gpus,
    num_epochs=FLAGS.num_epochs)

  # train the model
  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn,
    max_steps=max_steps,
    hooks=hooks)
  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    throttle_secs=FLAGS.eval_throttle_secs)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  FLAGS = parse_utils.parse_args()
  main()

