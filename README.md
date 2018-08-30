> ![@老徐](http://mirrors.softproject.net/avatar.png)Eric Xu
>
> Thursday, 30 August 2018

# TFLearn U-Net Starter

> https://www.kaggle.com/digdig/tflearn-u-net-starter/notebook

Quick and dirty kernel shows how to get started on segmenting nuclei using a neural network in TFLearn/Tensorflow.

Forked from Keras version: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277/notebook

# Paper

U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>

# Data

Using data from [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)

# Code 

IPython Notebook [TFLearn U-Net Starter](kernel.ipynb)

## Loss: What difference of binary_crossentropy between Keras and TFLearn

> If the last conv_2d's activity function is 'sigmoid'

Keras binary_crossentropy: loss = sigmoid(x)

```python
x = sigmoid(x)  # the last conv_2d
def binary_crossentropy(x):
    x = ~sigmoid(x) # undo sigmod(x), transform back to logits
    return tf.nn.sigmoid_cross_entropy_with_logits(x)
```

TFLearn binary_crossentropy: loss = sigmoid(sigmoid(x)) = always 0.693. it's wrong!!!

```python
x = sigmoid(x)  # the last conv_2d
def binary_crossentropy(x):
    return tf.nn.sigmoid_cross_entropy_with_logits(x)
```

should be
```python
x = linear(x)  # the last conv_2d
def binary_crossentropy(x):
    return tf.nn.sigmoid_cross_entropy_with_logits(x)
```
## Metric

```python
# Define IoU metric
def mean_iou_accuracy_op(y_pred, y_true, x):
    with tf.name_scope('Accuracy'):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_tmp = tf.to_int32(y_pred > 0.5)
            score, update_op = tf.metrics.mean_iou(y_true, y_pred_tmp, 2)
            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                score = tf.identity(score)
            prec.append(score)
        acc = tf.reduce_mean(tf.stack(prec), axis=0, name='mean_iou')
    return acc
```
