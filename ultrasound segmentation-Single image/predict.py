################################################################
# libraries
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers
import numpy as np
from PIL import Image
import sys
import logging

################################################################
# disable tensorflow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

################################################################
# read and process image
def get_original_and_transformed_image(image_path):
    # network input dimensions
    height, width = [128, 128]
    
    # read, resize and convert to grayscale
    test_image = Image.open(image_path).convert('L')
    test_image = test_image.resize((width, height))
    test_image = np.array(test_image, np.uint8)

    # normalize image
    test_image_transformed = test_image.astype(np.float32) * (1./255) - 0.5

    # add third channel
    test_image_transformed = test_image_transformed.reshape([height, width, 1])

    # convert to a batch of 1 image
    test_image_transformed = np.asarray([test_image_transformed])

    # return both oritinal and transformed images
    return test_image, test_image_transformed

################################################################
# model definition
def inference(data, is_training, batch_norm, scope=''):
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

################################################################
# main function

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        sys.exit('\nFull path to input image is not given.\n')

    if len(sys.argv) > 2:
        sys.exit('\n\nOnly one argument (full path to input image) should be given.\n')

    try:
        # read input image
        image_path = sys.argv[1]
        test_image, test_image_transformed = get_original_and_transformed_image(image_path)

        # load model
        sess = tf.Session()
        logits = inference(test_image_transformed, is_training=False, batch_norm=True)
        pred = tf.argmax(logits, dimension=3)[0]
        saver = tf.train.Saver(tf.all_variables())
        ckpt_path = "model_checkpoint/"
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.initialize_local_variables()) 

        # predict on test image
        predicted_mask = sess.run(pred)
        predicted_mask[predicted_mask == 1] = 255
        sess.close()

        # save predicted mask to image
        output_dir = "model_output/"
        Image.fromarray(test_image.astype('uint8')).save(output_dir + "test_image.png")
        Image.fromarray(predicted_mask.astype('uint8')).save(output_dir + "test_mask.png")
        
    except Exception as e:
        print(e)

