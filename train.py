import os
import numpy as np
from tensorflow import keras

from absl import app, flags, logging
from absl.flags import FLAGS

import utils
from datagen import BatchGenerator
import loss
import callbacks
import reader
import config
import datagen


"""
This script trains a model on triplets.
Example usage: 
    python train.py --save_to progress/test --num_epochs 10 --batch_size 8 --model mobilenet_v2 --input_size 224,224 --jitter --overwrite
"""

flags.DEFINE_string('save_to', config.SAVE_TO_DEFAULT, 'directory to save checkpoints and logs')
flags.DEFINE_boolean('overwrite', False, 'Overwrite given save path')
flags.DEFINE_string('from_ckpt', None, 'path to continue training on checkpoint')
flags.DEFINE_boolean('jitter', config.JITTER_DEFAULT, 'Apply image augmentation')
flags.DEFINE_integer('batch_size', config.BATCH_SIZE_DEFAULT, 'batch size')
flags.DEFINE_list('input_size', config.INPUT_SIZE_DEFAULT, 'input size in (width, height) format')
flags.DEFINE_integer('num_epochs', config.NUM_EPOCHS_DEFAULT, 'number of epochs')
flags.DEFINE_string('model', config.MODEL_DEFAULT, 'integer model type - %s'%str(config.MODELS.keys()))
flags.DEFINE_list('loss_weights',[0, 0], 'loss weights size in (w_dimension, w_orientation, w_confidence) format') # TODO configure

def main(_argv):
    assert not ((FLAGS.overwrite) and (FLAGS.from_ckpt is not None))
    input_size = list(map(int, FLAGS.input_size)) # (width, height)
    input_shape = (input_size[1], input_size[0], 3)
    loss_weights = list(map(float, FLAGS.loss_weights))

    # Load data
    logging.info("Loading data")
    data_reader = reader.DataReader()

    # Define batch generators
    logging.info("Creating batch generators")
    traingen = datagen.BatchGenerator(
        data_reader, 
        batch_size=FLAGS.batch_size,
        input_size = input_size,
        jitter = FLAGS.jitter,
        mode = 'train',
        # TODO Add arguments to BatchGenerator
        )
    valgen = datagen.BatchGenerator(
        data_reader, 
        batch_size=FLAGS.batch_size,
        input_size = input_size,
        jitter = False,
        mode = 'val',
        # TODO Add arguments to BatchGenerator
        )

    # Prepare network
    logging.info("Constructing model")
    model = config.MODELS[FLAGS.model](input_shape=input_shape) # TODO Add arguments to constructor
    
    # Setup and compile model
    model.compile(optimizer = 'adam', 
                loss='mse', # TODO change accordingly
                )
    
    # logging.info("Compiled model with loss weights:%s"%str(loss_weights)) # TODO Comment in if have loss weights
    model.summary()

    if FLAGS.from_ckpt is not None:
        logging.info("Loading weights from %s"%FLAGS.from_ckpt)
        model.load_weights(FLAGS.from_ckpt)

    logging.info("Genrating callbacks")
    train_callbacks = callbacks.get(directory = FLAGS.save_to, overwrite = FLAGS.overwrite)
    
    cfg = config.save(
        FLAGS.save_to, 
        model = FLAGS.model, 
        input_size = input_size, 
        jitter = FLAGS.jitter,
        batch_size = FLAGS.batch_size,
        num_epochs = FLAGS.num_epochs,
        # TODO Add arguments to config
        )

    logging.info("Saving config : %s"%str(cfg))
    logging.info("Starting training")
    model.fit(traingen,
                epochs=FLAGS.num_epochs,
                verbose=1,
                validation_data=valgen,
                callbacks=train_callbacks,
                workers = 8,
                max_queue_size=3)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass