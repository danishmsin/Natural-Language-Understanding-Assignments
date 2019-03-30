
## IWSLT English-Hindi

Train: 133K examples, vocab=vocab.(hi|en), train=train.(hi|en)
dev=tst2012.(hi|en),
test=tst2013.(hi|en), 

***Training details***. We train 2-layer LSTMs of 512 units with bidirectional
encoder (i.e., 1 bidirectional layers for the encoder), embedding dim
is 512. LuongAttention (scale=True) is used together with dropout keep_prob of
0.8. All parameters are uniformly. We use SGD with learning rate 1.0 as follows:
train for 12K steps (~ 12 epochs); after 8K steps, we start halving learning
rate every 1K step.



## WMT English-German

Train: 4.5M examples, vocab=vocab.bpe.32000.(de|en),
train=train.tok.clean.bpe.32000.(de|en), dev=newstest2013.tok.bpe.32000.(de|en),
test=newstest2015.tok.bpe.32000.(de|en),

***Training details***. Our training hyperparameters are similar to the
English-Hindi experiments except for the following details. 
(32K operations). We train 4-layer LSTMs of 1024 units with bidirectional
encoder (i.e., 2 bidirectional layers for the encoder), embedding dim
is 1024. We train for 350K steps (~ 10 epochs); after 170K steps, we start
halving learning rate every 17K step.


## Standard HParams

We have provided
[a set of standard hparams](nmt/standard_hparams/)
for using pre-trained checkpoint for inference or training NMT architectures
used in the Benchmark.

We will use the WMT16 German-English data, you can download the data by the
following command.

```
nmt/scripts/wmt16_en_de.sh /tmp/wmt16
```
