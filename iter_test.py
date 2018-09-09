import tensorflow as tf

# Create the example dataset
dataset_train = tf.data.Dataset.from_tensors(tf.constant('train')).repeat()
dataset_train = tf.data.Dataset.zip((dataset_train, tf.data.Dataset.range(9)))
dataset_test = tf.data.Dataset.from_tensors(tf.constant('test')).repeat()
dataset_test = tf.data.Dataset.zip((dataset_test, tf.data.Dataset.range(3)))


def initializable_iterator():
    '''
    Starts from the beginning every time
    :return:
    '''

    iterator = dataset_train.make_initializable_iterator()
    next_element = iterator.get_next()

    init_training = iterator.make_initializer(dataset_train)
    init_testing = iterator.make_initializer(dataset_test)

    with tf.Session() as sess:

        sess.run(init_training)
        for _ in range(3):
            s,i = sess.run(next_element)
            print('%s_%i' % (s.decode('utf-8'), i))

        sess.run(init_testing)
        for _ in range(3):
            s, i = sess.run(next_element)
            print('%s_%i' % (s.decode('utf-8'), i))

        sess.run(init_training)
        for _ in range(3):
            s, i = sess.run(next_element)
            print('%s_%i' % (s.decode('utf-8'), i))

def generic_iterator():

    iter_handle = tf.placeholder(tf.string, shape= [], name= 'iterator_handle')
    iterator : tf.data.Iterator = tf.data.Iterator.from_string_handle(iter_handle, dataset_train.output_types, dataset_train.output_shapes)
    next_element = iterator.get_next()

    train_iter_handle = dataset_train.make_one_shot_iterator().string_handle()
    test_iter = dataset_test.make_initializable_iterator()
    test_iter_handle = test_iter.string_handle()

    with tf.train.MonitoredTrainingSession() as sess:

        train_h, test_h = sess.run([train_iter_handle, test_iter_handle])

        while not sess.should_stop():
            for _ in range(3):
                s,i = sess.run(next_element, feed_dict= {iter_handle : train_h})
                print('%s_%i' % (s.decode('utf-8'), i))

            sess.run(test_iter.initializer)
            for _ in range(3):
                s,i = sess.run(next_element, feed_dict= {iter_handle : test_h})
                print('%s_%i' % (s.decode('utf-8'), i))

#initializable_iterator()
generic_iterator()