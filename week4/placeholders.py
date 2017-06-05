import tensorflow as tf

def build_graph():
    # Term n of a fibonacci sequence is the sum of last 2 terms.

    n_1 = tf.placeholder(tf.int32) # (n-1)th term
    n_2 = tf.placeholder(tf.int32) # (n-2)th term

    n = tf.add(n_1, n_2)

    return n_1, n_2, n

def run():
    sess = tf.Session()

    n_1, n_2, n = build_graph()

    # First 2 terms are 0 and 1.
    n_2_val = 0
    n_1_val = 1
    for i in range (3, 10):
        result = sess.run(n, feed_dict={n_2: n_2_val, n_1: n_1_val})
        print('Term {} is {}'.format(i, result))
        n_2_val = n_1_val
        n_1_val = result

def main():
    run()

if __name__ == "__main__":
    main()