
# first download the gpt_2 code
# !git clone https://github.com/nshepperd/gpt-2.git
# %pwd
# %cd gpt_2
# !pip3 install -r requirements.txt
# !python download_model.py 1558M
# !export PYTHONIOENCODING=UTF-8

import sys

sys.path.append("/gpt_2/src")
import os

os.environ['PYTHONIOENCODING'] = 'UTF-8'


import json
from gpt_2.src import model, sample, encoder, generate_unconditional_samples, interactive_conditional_samples

import os
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class GPT2:
    def __init__(
            self,
            model_name='345M',
            seed=None,
            nsamples=1,
            batch_size=1,
            length=None,
            temperature=1,
            top_k=0,
            raw_text="",
    ):
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        self.nsamples = nsamples
        self.batch_size = batch_size

        self.enc = encoder.get_encoder(model_name, 'gpt_2\\models')
        hparams = model.default_hparams()
        with open(os.path.join('gpt_2\\models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        self.sess = tf.compat.v1.Session()
        self.sess.__enter__()

        self.context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.compat.v1.train.Saver()
        self.ckpt = tf.compat.v1.train.latest_checkpoint(os.path.join('gpt_2\\models', model_name))
        saver.restore(self.sess, self.ckpt)

    def generate_conditional(self, raw_text):
        context_tokens = self.enc.encode(raw_text)
        generated = 0
        for _ in range(self.nsamples // self.batch_size):
            out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(self.batch_size)]
            })[:, len(context_tokens):]
            for i in range(self.batch_size):
                generated += 1
                text = self.enc.decode(out[i])
                return text


gpt2 = GPT2(model_name="1558M")

result = gpt2.generate_conditional(raw_text="Can you tell me something about music?")

print(result)