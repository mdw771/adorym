import tensorflow as tf




global_step = tf.Variable(0, trainable=False, name='global_step')

out = global_step * 2

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in range(10):

    sess.run(out)
    print(tf.train.global_step(sess, global_step))