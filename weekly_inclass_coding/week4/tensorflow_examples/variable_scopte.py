import tensorflow as tf

def create_var(sess, var_name, value):
    """Mimics 'sum = <value>' in Python.
    Creates a new variable and initializes it"""
    var = tf.get_variable(var_name, [], dtype=tf.int32)
    sess.run(tf.assign(var, value))
    return var

def add_to_var(sess, var, value):
    """Mimics 'sum += <value>' in Python
    Reads an existing variable and adds a value to it"""
    value = sess.run(var) + value
    sess.run(tf.assign(var, value))

def correct_loop(sess):
    sum = create_var(sess, 'sum', 0) # Creates variable 'correct_loop/sum'
    for i in range(10):
        add_to_var(sess, sum, i)
    return sess.run(sum)

def wrong_loop(sess):
    sum = create_var(sess, 'sum', 0) # Creates variable 'wrong_loop/sum'
    for i in range(10):
        add_to_var(sess, sum, i)
        tf.get_variable_scope().reuse_variables() # Remove safeguard
        sum = create_var(sess, 'sum', 15) # Resets 'wrong_loop/sum'!
    return sess.run(sum)

def run():
    sess = tf.Session()
    with tf.variable_scope('correct_loop'):
       print(correct_loop(sess)) # Prints 45
    with tf.variable_scope('wrong_loop'):
       print(wrong_loop(sess)) # Prints 15

    sess.close()

def main():
    run()

if __name__ == "__main__":
    main()