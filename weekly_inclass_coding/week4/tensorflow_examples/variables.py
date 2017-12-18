import tensorflow as tf


def returnsometensor(x1):
    res = tf.add(x1, tf.constant(5.))

    return res


def run():
    sess = tf.Session()

    x1 = tf.constant(0.0, dtype="float", shape=None)

    res = returnsometensor(x1)

    somevariable = tf.get_variable('somevariable', [], dtype=tf.float32)  # [] - a scalar

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess.run(init)  # run variable initializer!!!!

    # compute the value of res tensor
    value = sess.run(res, feed_dict={x1: 3.})
    # assign the value of res to somevariable
    sess.run(somevariable.assign(value))

    currentvalues = sess.run([somevariable])
    print(' ')
    print(' ')
    print('original value of somevariable', currentvalues)

    # one saves a session! not variables
    saver.save(sess, 'checkpoints/tester.ckpt', global_step=0)  # global_step is a step counter (e.g. train epochs)

    # recompute result, reassign to variable
    value = sess.run(res, feed_dict={x1: 0.})
    sess.run(somevariable.assign(value))

    currentvalues = sess.run(somevariable, feed_dict={x1: 0.})
    print('value of somevariable after new computation', currentvalues)

    # reassign the tensor res, when evaluated, it will have value -3
    # res=tf.constant(-3.0,dtype="float", shape=None)
    # somevariable = tf.constant(-3.0,dtype="float", shape=None) # one way to prevent a variable to be overwritten upon restore is to overwrite it with a non-restorable object, here a standard tensor



    # restore - this affects somevariable as long as you did not change it to a tensor like with the command above
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    # but restore will not affect tensor res

    print('value after restore: somevariable', sess.run(somevariable))
    print(' ')

    print('value after restore: result tensor (used to feed variable): sess.run(res) ', sess.run(res))
    print('NOTE!! result tensor: sess.run(res) RECOMPUTES THE TENSOR! it does NOT simply print it')

    # sess.run(res.assign(somevariable)) # you cannot assign a value to a tensor!! this will create an error
    # you can do this ONLY with variables - which are mutable

    # with a variable v: one can define it, then run sess.run(v.assign(xvalue)) so that it gets a value xvalue
    # any later sess.run(v) or any call to v when used in a graph will use the value xvalue


    # this also explains why we should use placeholders for feeding in data:


if __name__ == "__main__":
    run()