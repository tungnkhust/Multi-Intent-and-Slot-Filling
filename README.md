# MixATIS
## 1 No attention + Adam + ReduceLROnPlateau
- hyperarameter
```python
word_embed_size=64,
hidden_size=64,
n_rnn_layers=2,
 dropout=0.4,
mode='all',

# train
batch_size=128,
max_seq_len=80,
early_stop_max_epochs=3,
lr=0.01,
num_epochs=50,
checkpoint=False,
thresh=0.7,
clip=5
```

### No init xavier uniform
```
f1: 0.6754346348096347    precision: 0.9390059593184594    recall: 0.6754346348096347
accuracy: 0.6485570443662911    sub accuracy : 0.0    hamming loss: 0.03900049603174603
```

### init with xavier uniform
```
f1: 0.7251699057937568    precision: 0.9548448946886448    recall: 0.7251699057937568
accuracy: 0.7036697205470184    sub accuracy : 0.0234375    hamming loss: 0.03292410714285714
```
 `====>` Conclusion: init with xavier uniform have better performance. -> use init xavier

 ## 2 Attention Luong general
```
f1: 0.8028251415659504    precision: 0.8983713673224244    recall: 0.8028251415659504
accuracy: 0.7420963763900466    sub accuracy : 0.03125    hamming loss: 0.030567956349206348
```
`====>` Conclusion: use attention have better performace . -> attention.
 ## 3 Attention Luong general + init xavier + OneCycleLR + clipping=5
```
f1: 0.8528728936759729    precision: 0.9007790973324062    recall: 0.8528728936759729
accuracy: 0.7866473302924178    sub accuracy : 0.140625    hamming loss: 0.025669642857142856
```
`====>` Conclusion: use OneCycleLR have better performace -> OneCycleLR.

 ## 3 Attention Luong general + init xavier + OneCycleLR + no clipping
 ```
 f1: 0.8405189033291056    precision: 0.8751977093831413    recall: 0.8405189033291056
accuracy: 0.7561899287811754    sub accuracy : 0.0625    hamming loss: 0.029637896825396817
```
`====>` Conclusion: use clipping have better performace -> clipping=5.

## Best performace Attention Luong general + init xavier + OneCycleLR + clipping


## 2 Attention Luong general + init xavier + OneCycleLR + clipping
- hyperarameter
```python
word_embed_size=64,
hidden_size=64,
n_rnn_layers=3,
 dropout=0.4,
mode='all',

# train
batch_size=128,
max_seq_len=80,
early_stop_max_epochs=3,
lr=0.01,
num_epochs=50,
checkpoint=False,
thresh=0.7,
clip=5
```

### n_rnn_layers=3
```
f1: 0.8629090986627753    precision: 0.9004418802959705    recall: 0.8629090986627753
accuracy: 0.7955604809706976    sub accuracy : 0.171875    hamming loss: 0.024615575396825396
```
### n_rnn_layers=4 
f1: 0.8603625719664325    precision: 0.8836534737729592    recall: 0.8603625719664325
accuracy: 0.7807541092727577    sub accuracy : 0.109375    hamming loss: 0.026723710317460316


# save train state to continue train.