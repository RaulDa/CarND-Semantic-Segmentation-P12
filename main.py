import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import scipy.misc
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Define session as global
sess = tf.Session()
# Define logits tensor as global
logits = tf.zeros([160*576,2])
# Define global placeholders
keep_prob = tf.placeholder(tf.float32)
input_image = tf.placeholder(tf.float32, (None, None, None, 3))
# Image shape
image_shape = (160, 576)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # Names of tensors to be extracted from network
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    op = graph.get_operations()
    [print(m.values()) for m in op][1]

    # Get tensors
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    print(tf.trainable_variables())
    
    return input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Scale layers 4 and 3 to compensate vgg scale
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.0001, name='vgg_layer4_out')
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.01, name='vgg_layer3_out')

    # Apply transposed convolutions to layers 4 and 7
    vgg_layer7_out = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, kernel_size=4, strides=(2, 2), padding='same')
    vgg_layer7_out = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, kernel_size=4, strides=(2, 2), padding='same')
    vgg_layer4_out = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, kernel_size=4, strides=(2, 2), padding='same')
    # Sum both layers
    output = tf.add(vgg_layer7_out, vgg_layer4_out)
    # Transposed convolution to output and regularizer applied
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=16, strides=(8, 8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Transposed convolution to layer 3 and addition to output
    vgg_layer3_out = tf.layers.conv2d_transpose(vgg_layer3_out, num_classes, kernel_size=8, strides=(8, 8), padding='same')
    output = tf.add(output, vgg_layer3_out)

    return output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape logits and layers
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Build optimizer and add regularization losses to total loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_fn = cross_entropy_loss + tf.reduce_sum(reg_ws)
    train_op = optimizer.minimize(loss_fn)

    return logits, train_op, loss_fn
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Initialization of variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Training
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            [tr_operation, cr_entropy_loss] = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob:0.5, learning_rate:0.00075})
        print("EPOCH {} ...".format(epoch+1))
        print("Loss: {}".format(cr_entropy_loss))
    pass
tests.test_train_nn(train_nn)

def detect_video(img):
    # Apply logits to video images and apply calculated mask to each of them
    image = scipy.misc.imresize(img, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob:1.0, input_image:[image]})
    im_softmax = im_softmax[0][:,1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0,255,0,127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return np.array(street_im)

def run():
    num_classes = 2
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 1#15
    batch_size = 1#20
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.float32, (None, None, None, num_classes))

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # Build NN using load_vgg, layers, and optimize function
    global input_image, keep_prob
    input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
    output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
    global logits
    logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

    # Train NN using the train_nn function
    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

    # Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

    # Apply the trained model to a video
    project_output = 'project_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(detect_video)
    project_clip.write_videofile(project_output, audio=False)

    sess.close()

if __name__ == '__main__':
    run()
