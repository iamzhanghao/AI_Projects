import tensorflow as tf
import math

def build_graph():
    x = tf.placeholder(tf.float32)
    y = tf.sin(x)
    report_y = tf.summary.scalar('y', y)

    z = tf.square(y)
    report_z = tf.summary.scalar('z', z)

    return report_y, x

def run():
    sess = tf.Session()

    report, x = build_graph()

    writer = tf.summary.FileWriter('./log', sess.graph)

    input = 0
    step = 0.1
    number_of_steps = int(math.ceil((4*22/7) / step)) # domain, 4 pi
    for i in range(number_of_steps):
        print('i: {}'.format(i), end='\r')
        input += 0.1
        # Run only tensor 'report_y', not 'report_z'.
        summ = sess.run(report, feed_dict={x: input})
        writer.add_summary(summ, i)

    print('\n')
    sess.close()

def main():
    run()

if __name__ == "__main__":
    main()