import math
import sys

from PIL import Image

sys.path.insert(0, '.')
sys.path.insert(0, '../imagenetdata')

from week6.mon.getimagenetclasses import *
from week6.mon.alexnet import *


def preproc_py2(imname, shorterside):
    pilimg = Image.open(imname)
    w, h = pilimg.size

    print((w, h))

    if w > h:
        longerside = np.int32(math.floor(float(shorterside) / float(h) * w))
        neww = longerside
        newh = shorterside
    else:
        longerside = np.int32(math.floor(float(shorterside) / float(w) * h))
        newh = longerside
        neww = shorterside
    resimg = pilimg.resize((neww, newh))

    # im = np.array(resimg, dtype=np.float32)
    im = np.array(resimg)

    return im


def cropped_center(im, hsize, wsize):
    h = im.shape[0]
    w = im.shape[1]

    cim = im[int((h - hsize) / 2):int((h - hsize) / 2 + hsize), int((w - wsize) / 2):int((w - wsize) / 2 + wsize), :]

    return cim


def preproc(image):
    '''
  filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpg"))

  # Read an entire image file which is required since they're JPEGs, if the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)
  
  '''

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    new_shorter_edge = tf.cast(227, tf.int32)

    def _compute_longer_edge(height, width, new_shorter_edge):
        return tf.cast(width * new_shorter_edge / height, tf.int32)

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
        lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
    )

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    resimg = tf.image.resize_images(
        image,
        new_height_and_width
        # tf.stack(new_height_and_width) #,
        # tf.concat(new_height_and_width)
        # method=ResizeMethod.BILINEAR,
        # align_corners=False
    )
    resimg = tf.expand_dims(resimg, 0)
    # image = tf.subtract(image, 110.0)

    return resimg


def getout():
    batchsize = 1
    is_training = False
    num_classes = 1000
    keep_prob = 1.
    skip_layer = []

    x = tf.placeholder(tf.float32, [batchsize, 227, 227, 3])
    net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')

    out = net.fc8

    return out, x, net


def save_img(image, show=False):
    if image.ndim == 4:
        img = Image.fromarray(image[0].astype(np.uint8), "RGB")
    else:
        img = Image.fromarray(image.astype(np.uint8), "RGB")
    img.save('C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\modified_img0.png')
    if show:
        img.show()


def run2():
    stepsize = 20.0
    desired_label = 949
    imname = 'C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\mrshout2.jpg'

    imagenet_mean = np.array([104., 117., 123.], dtype=np.float32)
    cls = get_classes()

    sess = tf.Session()

    out, x, net = getout()
    init = tf.global_variables_initializer()
    sess.run(init)
    net.load_initial_weights(sess)

    image = preproc_py2(imname, 250)

    print(image.shape)
    print(imname)

    # convert grey to color
    if (image.ndim < 3):
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), 2)

    # dump alpha channel if it exists
    if (image.shape[2] > 3):
        image = image[:, :, 0:3]

    # here need to average over 5 crops instead of one
    image_crop = cropped_center(image, 227, 227)

    image_crop = image_crop[:, :, [2, 1, 0]]  # RGB to BGR
    image_crop = image_crop - imagenet_mean
    image_crop = np.expand_dims(image_crop, 0)
    save_img(image_crop)

    # run initial classification
    predict_values = sess.run(out, feed_dict={x: image_crop})


    original_label = np.argmax(predict_values)

    print('at start: classindex: ' + str(original_label) + '\nclasslabel: ' + cls[
        np.argmax(predict_values)] + '\nscore:' + str(np.max(predict_values)))

    print(predict_values[0, desired_label], predict_values[0, original_label])


    ## attempt 1
    # sess.close()

    # current_label = original_label
    # tf_img = tf.Variable(image_crop,name = 'tf_img')
    # score = tf.Variable(predict_values[0,desired_label],name="predict_values")
    #
    #
    # optimizer = tf.train.GradientDescentOptimizer(20)
    # train = optimizer.minimize(-score)
    #
    # init = tf.initialize_all_variables()
    #
    # with tf.Session() as session:
    #     session.run(init)
    #     step = 0
    #     while(current_label != desired_label):
    #         step += 1
    #         session.run(train)
    #         print("step"+ str(step), "score "+ str(session.run(score)),end="")
    #         predict_values = sess.run(out, feed_dict={x: session.run(tf_img)})
    #         print(" prediction:" + str(np.argmax(predict_values)))

    ## attempt 2

    current_label = original_label
    gradient = tf.gradients(tf.slice(out,[0,1],[0,desired_label]),x)

    while current_label != desired_label:
        gradient_val = sess.run(gradient, feed_dict={x: image_crop})
        image_crop += np.tensordot(stepsize,gradient_val)
        print(gradient_val)
        break


if __name__ == '__main__':
    run2()
    # m=np.load('./ilsvrc_2012_mean.npy')
    # print(np.mean(np.mean(m,2),1))
