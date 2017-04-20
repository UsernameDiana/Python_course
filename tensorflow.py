import tensorflow as tf

# setup
# x = the values/pixels, w = weight, b = bias(angle of freedoom)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
# place holder for x, tensor = multi dimentional array[]

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))



# model
y = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 784]),w) +b)
# nn = neuron network, part of tensorflow
# matmul = matrix multiplying
# reshape = flattering image to vektor

# placeholder for correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

# Loss function
loss_function = -tf.reduce_sum(y_ * tf.log(y))

# % of correct (images)answers found
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



# Training = computing variables W and b
# Training step
optimizer = tr.train.GradientDescentOptimizer(0.003) # learning rate
# compute gradiant, feed it with small learning rate number
train_steps = optimizer.minimize(loss_function)
# after modifying, we apply for the next training step


# To compute value in Tenserflow, we need to define a session to run on a node with data
sess = tf.Session()
sess.run(init)


# Training loop
for i in range(1000)
    # Load batch of images and correct ansvers
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = { x: batch_x, y_: batch_y }

    # train
    sess.run(train_steps, feed_dict = train_data)
    # feed_dictionary with keys of x and y, the plaveholders

    # success ?
    a,c = sess.run ([accuracy, loss_function], feed_dict = train_data)

    # success on test data ?
    test_data = { x: mnist.test.images, y_: mnist.test.labels}
    a,c = sess.run ([accuracy, loss_function], feed = test_data)
