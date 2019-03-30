Embedding
Given the categorical nature of words, the model must first look up the source and target embeddings to retrieve the corresponding word representations. For this embedding layer to work, a vocabulary is first chosen for each language. Usually, a vocabulary size V is selected, and only the most frequent V words are treated as unique. All other words are converted to an "unknown" token and all get the same embedding. The embedding weights, one set per language, are usually learned during training.

# Embedding
embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_encoder, encoder_inputs)
Similarly, we can build embedding_decoder and decoder_emb_inp. Note that one can choose to initialize embedding weights with pretrained word representations such as word2vec or Glove vectors. In general, given a large amount of training data we can learn these embeddings from scratch.

Encoder
Once retrieved, the word embeddings are then fed as input into the main network, which consists of two multi-layer RNNs – an encoder for the source language and a decoder for the target language. These two RNNs, in principle, can share the same weights; however, in practice, we often use two different RNN parameters (such models do a better job when fitting large training datasets). The encoder RNN uses zero vectors as its starting states and is built as follows:

# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=source_sequence_length, time_major=True)
Note that sentences have different lengths to avoid wasting computation, we tell dynamic_rnn the exact source sentence lengths through source_sequence_length. Since our input is time major, we set time_major=True. Here, we build only a single layer LSTM, encoder_cell. We will describe how to build multi-layer LSTMs, add dropout, and use attention in a later section.

Decoder
The decoder also needs to have access to the source information, and one simple way to achieve that is to initialize it with the last hidden state of the encoder, encoder_state. In Figure 2, we pass the hidden state at the source word "student" to the decoder side.

# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, decoder_lengths, time_major=True)
# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output
Here, the core part of this code is the BasicDecoder object, decoder, which receives decoder_cell (similar to encoder_cell), a helper, and the previous encoder_state as inputs. By separating out decoders and helpers, we can reuse different codebases, e.g., TrainingHelper can be substituted with GreedyEmbeddingHelper to do greedy decoding. See more in helper.py.

Lastly, we haven't mentioned projection_layer which is a dense matrix to turn the top hidden states to logit vectors of dimension V. We illustrate this process at the top of Figure 2.

projection_layer = layers_core.Dense(
    tgt_vocab_size, use_bias=False)
Loss
Given the logits above, we are now ready to compute our training loss:

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) /
    batch_size)
Here, target_weights is a zero-one matrix of the same size as decoder_outputs. It masks padding positions outside of the target sequence lengths with values 0.

Important note: It's worth pointing out that we divide the loss by batch_size, so our hyperparameters are "invariant" to batch_size. Some people divide the loss by (batch_size * num_time_steps), which plays down the errors made on short sentences. More subtly, our hyperparameters (applied to the former way) can't be used for the latter way. For example, if both approaches use SGD with a learning of 1.0, the latter approach effectively uses a much smaller learning rate of 1 / num_time_steps.

Gradient computation & optimization
We have now defined the forward pass of our NMT model. Computing the backpropagation pass is just a matter of a few lines of code:

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm)
One of the important steps in training RNNs is gradient clipping. Here, we clip by the global norm. The max value, max_gradient_norm, is often set to a value like 5 or 1. The last step is selecting the optimizer. The Adam optimizer is a common choice. We also select a learning rate. The value of learning_rate can is usually in the range 0.0001 to 0.001; and can be set to decrease as training progresses.

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))
In our own experiments, we use standard SGD (tf.train.GradientDescentOptimizer) with a decreasing learning rate schedule, which yields better performance. See the benchmarks.

Hands-on – Let's train an NMT model
Let's train our very first NMT model, translating from Vietnamese to English! The entry point of our code is nmt.py.

We will use a small-scale parallel corpus of TED talks (133K training examples) for this exercise.

Run the following command to download the data for training NMT model:code/nmt/scripts/download_iwslt15.sh /tmp/nmt_data

Run the following command to start the training:

mkdir /tmp/nmt_model
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
The above command trains a 2-layer LSTM seq2seq model with 128-dim hidden units and embeddings for 12 epochs. We use a dropout value of 0.2 (keep probability 0.8). If no error, we should see logs similar to the below with decreasing perplexity values as we train.

# First evaluation, global step 0
  eval dev: perplexity 17193.66
  eval test: perplexity 17193.27
# Start epoch 0, step 0, lr 1, Tue Apr 25 23:17:41 2017
  sample train data:
    src_reverse: </s> </s> Ði?u dó , di nhiên , là câu chuy?n trích ra t? h?c thuy?t c?a Karl Marx .
    ref: That , of course , was the <unk> distilled from the theories of Karl Marx . </s> </s> </s>
  epoch 0 step 100 lr 1 step-time 0.89s wps 5.78K ppl 1568.62 bleu 0.00
  epoch 0 step 200 lr 1 step-time 0.94s wps 5.91K ppl 524.11 bleu 0.00
  epoch 0 step 300 lr 1 step-time 0.96s wps 5.80K ppl 340.05 bleu 0.00
  epoch 0 step 400 lr 1 step-time 1.02s wps 6.06K ppl 277.61 bleu 0.00
  epoch 0 step 500 lr 1 step-time 0.95s wps 5.89K ppl 205.85 bleu 0.00
See train.py for more details.

We can start Tensorboard to view the summary of the model during training:

tensorboard --port 22222 --logdir /tmp/nmt_model/
Training the reverse direction from English and German can be done simply by changing:
--src=en --tgt=de