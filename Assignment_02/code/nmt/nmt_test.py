
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from . import inference
from . import nmt
from . import train


def _update_flags(flags, test_name):
 
  flags.num_train_steps = 100
  flags.steps_per_stats = 5
  flags.src = "en"
  flags.tgt = "vi"
  flags.train_prefix = ("nmt/testdata/"
                        "iwslt15.tst2013.100")
  flags.vocab_prefix = ("nmt/testdata/"
                        "iwslt15.vocab.100")
  flags.dev_prefix = ("nmt/testdata/"
                      "iwslt15.tst2013.100")
  flags.test_prefix = ("nmt/testdata/"
                       "iwslt15.tst2013.100")
  flags.out_dir = os.path.join(tf.test.get_temp_dir(), test_name)


class NMTTest(tf.test.TestCase):

  def testTrain(self):
   
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_test")

    default_hparams = nmt.create_hparams(FLAGS)

    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)


  def testTrainWithAvgCkpts(self):
   
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_test_avg_ckpts")
    FLAGS.avg_ckpts = True

    default_hparams = nmt.create_hparams(FLAGS)

    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)


  def testInference(self):

    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    _update_flags(FLAGS, "nmt_train_infer")

    
    FLAGS.num_train_steps = 1
    default_hparams = nmt.create_hparams(FLAGS)
    train_fn = train.train
    nmt.run_main(FLAGS, default_hparams, train_fn, None)

    
    FLAGS.inference_input_file = ("nmt/testdata/"
                                  "iwslt15.tst2013.100.en")
    FLAGS.inference_output_file = os.path.join(FLAGS.out_dir, "output")
    FLAGS.inference_ref_file = ("nmt/testdata/"
                                "iwslt15.tst2013.100.vi")

    default_hparams = nmt.create_hparams(FLAGS)

    inference_fn = inference.inference
    nmt.run_main(FLAGS, default_hparams, None, inference_fn)


if __name__ == "__main__":
  tf.test.main()
