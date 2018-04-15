import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob, math, cv2, time
import numpy as np
import sys
import re
import csv
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers
from scipy.misc import imresize

train_path='train/' #path where the training images are
test_path='test/' #path to where the testing images are
record_path='tfrecords/' #where the TFrecords will be saved
tensorb_path='events/'#where the tensorboard events will be saved
ckpt_path="checkpoints/"

height=128  #original size : 420*580
width=128
def process_image(img_file):
    img = cv2.imread(img_file,1)
    img = (cv2.resize(img,(height,width)).astype('uint8'))
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return img

def load_training():
    start = time.time()
    id_patient=[]
    X_train=[]
    Y_train=[]
    print('Loading training data')
    path = train_path+'*[0-9].tif'
    files = glob.glob(path)
    for im_file in files:
        id_patient.append(os.path.basename(im_file).split('_')[0])
        img=process_image(im_file)
        mask=os.path.splitext(im_file)[0]+'_mask.tif'
        X_train.append(img)
        imgmask= process_image(mask)
        Y_train.append(imgmask)
    end = time.time() - start
    print("Number of training examples : %.4r" % (len(X_train)))
    print("Time: %.2f seconds" % end)
    X_train=np.asarray(X_train)
    Y_train=np.asarray(Y_train)
    return X_train,Y_train,id_patient


def load_testing():
    start = time.time()
    test=[]
    idtest=[]
    print('Loading testing data')
    path = test_path+'*.tif'
    files = glob.glob(path)
    for im_file in files:
        img=process_image(im_file)
        test.append(img)
        idtest.append((im_file.split('/')[-1].split('.')[0]))
    end = time.time() - start
    print("Number of testing examples : %.4r" % (len(test)))
    print("Time: %.2f seconds" % end)
    test=np.asarray(test)
    return test,idtest

def show_image_and_mask(idx):
    """ Displays a training image and the mask. """
    im = X[idx]
    plt.subplot(121)
    plt.imshow(im,cmap="gray")

    mask = Y[idx]
    plt.subplot(122)
    plt.imshow(mask,cmap="gray")


X,Y,id_patient=load_training()
test,idtest=load_testing()

Xtrain=X[0:4552]
Ytrain=Y[0:4552]
Xvalid=X[4552:]
Yvalid=Y[4552:]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images,masks,name):
    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = 1

    filename = os.path.join(record_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        mask_raw=masks[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(rows),'width': _int64_feature(cols),'depth': _int64_feature(depth),'image_raw': _bytes_feature(image_raw),'mask_raw': _bytes_feature(mask_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def convert_topred(images,idtest,name):
    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = 1

    filename = os.path.join(record_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(rows),'width': _int64_feature(cols),'depth': _int64_feature(depth),'image_raw': _bytes_feature(image_raw),'idtest': _int64_feature(int(idtest[index]))}))
        writer.write(example.SerializeToString())
    writer.close()

convert_to(Xtrain,Ytrain,'trainrecord')
convert_to(Xvalid,Yvalid,'validrecord')
convert_topred(test,idtest,'testrecord')


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'image_raw': tf.FixedLenFeature([], tf.string),'mask_raw':tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [height,width])

    mask=tf.decode_raw(features['mask_raw'], tf.uint8)
    mask = tf.reshape(mask, [height,width])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  #Convert from [0, 255] -> [0, 1] floats.
    mask=tf.cast(mask, tf.float32) * (1. / 255)
    return image,mask

def read_and_decode_pred(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'image_raw': tf.FixedLenFeature([], tf.string),'idtest': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [height,width])

 # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    idtest = tf.cast(features['idtest'], tf.int32)
    return image,idtest

def image_distortions(image, distortions):
    distort_left_right_random = distortions[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    distort_up_down_random = distortions[1]
    mirror = tf.less(tf.stack([distort_up_down_random, 1.0, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    return image

def distorted_inputs(batch_size, num_epochs):
  #Construct distorted input for  training
    if not num_epochs: num_epochs = None
    filename =record_path+ 'trainrecord.tfrecords'

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    image,mask= read_and_decode(filename_queue)

    image= tf.reshape(image, [height,width,1])
    mask=tf.reshape(mask, [height,width,1])

    distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
    image = image_distortions(image, distortions)
    mask = image_distortions(mask, distortions)# image and mask will be distorted the same way

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    with tf.device('/cpu:0'):
        images, masks = tf.train.shuffle_batch([image,mask], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,allow_smaller_final_batch=False,min_after_dequeue=1000)
        # Ensures a minimum amount of shuffling of examples.

    tf.image_summary('image', images,max_images=2) #This will show the images on TensorBoard
    tf.image_summary('mask', masks,max_images=2)
    masks=tf.cast(masks, tf.int64)
    return images,masks

def inputs(train, batch_size, num_epochs):
  #same as the previous function but images are not distorted (for evaluation)
    if train==True:
        filename = record_path+'trainrecord.tfrecords'

    else:
        filename = record_path+'validrecord.tfrecords'

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image,mask= read_and_decode(filename_queue)

    image=tf.reshape(image, [height,width,1])
    mask=tf.reshape(mask, [height,width,1])

    with tf.device('/cpu:0'):
        images, masks = tf.train.batch([image,mask], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,allow_smaller_final_batch=False)
    tf.image_summary('image', images,max_images=2)
    tf.image_summary('mask', masks,max_images=2)
    masks=tf.cast(masks, tf.int64)
    return images,masks

def inputs_topred(batch_size, num_epochs):
    filename = record_path+'testrecord.tfrecords'
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image,idtest=read_and_decode_pred(filename_queue)
        image=tf.reshape(image, [height,width,1])
    with tf.device('/cpu:0'):
        images,idtests = tf.train.batch([image,idtest],batch_size=batch_size, num_threads=1,allow_smaller_final_batch=True,capacity=1000 + 3 * batch_size)
        return images,idtests


def inference(data,is_training,batch_norm,scope=''):
    prob=0.60
    with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=slim.batch_norm,
                     normalizer_params={'decay': 0.9997,'is_training':batch_norm,'updates_collections':None,
                     'trainable':is_training},
                      weights_initializer=slim.initializers.xavier_initializer()
                     ,weights_regularizer=slim.l2_regularizer(0.0005)):
        conv1 = slim.repeat(data, 2, slim.conv2d, 32, [3, 3], scope='conv1')
        conv1 = slim.dropout(conv1, prob,is_training=is_training,scope='dropout1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1') # 64
        conv2 = slim.repeat(pool1, 2, slim.conv2d, 64, [3, 3], scope='conv2')
        conv2 = slim.dropout(conv2, prob,is_training=is_training,scope='dropout2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2') # 32
        conv3 = slim.repeat(pool2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
        conv3 = slim.dropout(conv3, prob,is_training=is_training,scope='dropout3')
        pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3') # 16
        conv4 = slim.repeat(pool3, 2, slim.conv2d, 256, [3, 3], scope='conv4')
        conv4 = slim.dropout(conv4, prob,is_training=is_training,scope='dropout4')
        pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')#8
        conv5 = slim.repeat(pool4, 2, slim.conv2d, 512, [3, 3], scope='conv5')
        conv5 = slim.dropout(conv5, prob,is_training=is_training,scope='dropout5')

    with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], padding='SAME',
                     normalizer_fn=slim.batch_norm,
                     normalizer_params={'decay': 0.9997,'is_training':batch_norm,'updates_collections':None,
                    'trainable':is_training},
                     weights_initializer=slim.initializers.xavier_initializer(),
                     weights_regularizer=slim.l2_regularizer(0.0005)):
        deconv1=slim.conv2d_transpose(conv5,512,stride=2,kernel_size=2) #16
        deconv1 = slim.dropout(deconv1, prob,is_training=is_training,scope='d_dropout1')
        concat1=tf.concat(3,[conv4,deconv1],name='concat1')
        conv6 = slim.repeat(concat1, 2, slim.conv2d, 256, [3, 3], scope='conv6')
        conv6 = slim.dropout(conv6, prob,is_training=is_training,scope='dropout6')
        deconv2=slim.conv2d_transpose(conv6,256,stride=2,kernel_size=2) #32
        deconv2 = slim.dropout(deconv2, prob,is_training=is_training,scope='d_dropout2')
        concat2=tf.concat(3,[conv3,deconv2],name='concat2')
        conv7 = slim.repeat(concat2, 2, slim.conv2d,128, [3, 3], scope='conv7')
        conv7 = slim.dropout(conv7, prob,is_training=is_training,scope='dropout7')
        deconv3=slim.conv2d_transpose(conv7,128,stride=2,kernel_size=2) #64
        deconv3 = slim.dropout(deconv3, prob,is_training=is_training,scope='d_dropout3')
        concat3=tf.concat(3,[conv2,deconv3],name='concat3')
        conv8 = slim.repeat(concat3, 2, slim.conv2d, 64, [3, 3], scope='conv8')
        conv8 = slim.dropout(conv8, prob,is_training=is_training,scope='dropout8')
        deconv4=slim.conv2d_transpose(conv8,64,stride=2,kernel_size=2) #128
        deconv4 = slim.dropout(deconv4, prob,is_training=is_training,scope='d_dropout4')
        concat4=tf.concat(3,[conv1,deconv4],name='concat4')
        conv9 = slim.repeat(concat4, 2, slim.conv2d, 32, [3, 3], scope='conv9')
        conv9 = slim.dropout(conv9, prob,is_training=is_training,scope='dropout9')
        conv1x1=slim.conv2d(conv9, 2, [1, 1],  activation_fn=tf.nn.sigmoid,scope='conv1x1')
    return conv1x1

def dice_coef(y_true, y_pred):
    """Compute the mean(batch-wise) of dice coefficients"""
    md=tf.constant(0.0)
    y_true=tf.cast(y_true, tf.float32)
    y_true_f = slim.flatten(y_true)
    y_pred_f = slim.flatten(y_pred)
    for i in xrange(batch_size):
        union=tf.reduce_sum(y_true_f[i]) + tf.reduce_sum(y_pred_f[i])
        md = tf.cond(tf.equal(union,0.0), lambda: tf.add(md,1.0),
                 lambda: tf.add(md,tf.div(2.*tf.reduce_sum(tf.mul(y_true_f[i],y_pred_f[i])),union)))

    return tf.div(md,batch_size)

num_epochs=55
batch_size=24

def train():

  with tf.Graph().as_default():

    images, masks =distorted_inputs(batch_size=batch_size,num_epochs=num_epochs)
    masks=tf.reshape(masks,[batch_size,height,width], name=None)
    logits=inference(images,is_training=True,batch_norm=True)


    pred=tf.argmax(logits, dimension=3)
    pred=tf.reshape(pred, [batch_size,height,width,1])
    pred=tf.cast(pred, tf.float32)
    tf.image_summary('predicted', pred,max_images=2) # This will display predicted images in tensorboard

    class_weight = tf.constant([0.75,0.25])
    weighted_logits =tf.mul(logits, class_weight)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(weighted_logits, masks)

    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    dice=dice_coef(masks,pred)

    tf.scalar_summary('loss', loss)#tensorboard will keep track of loss and dice during training
    tf.scalar_summary('dice', dice)


    train_op=tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(loss)




    # Build the summary operation based on the TF collection of Summaries.
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    # Build an initialization operation to run below.
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())



    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.train.SummaryWriter(tensorb_path, sess.graph)
    try:
        step = 0
        while not coord.should_stop():


            start_time = time.time()
            _, loss_value,dice_value= sess.run([train_op, loss,dice])

            duration = time.time() - start_time
            if step % 10 == 0:

              #print('Step %d: loss = %.10f (%.3f sec) dice=%.2f' % (step, loss_value,duration,dice_value))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0 :
                saver.save(sess,ckpt_path+ 'model.ckpt',step) #makes a checkpoint every 1000 step
            if step % 100==0:
                print('Step %d: loss = %.5f dice=%.2f (%.3f sec) ' % (step, loss_value,dice_value,duration))
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

train()
