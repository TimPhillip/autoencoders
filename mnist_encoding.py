import tensorflow as tf
import argparse
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os

from autoencoders.autoencoder import Autoencoder, VariationalAutoencoder

def mnist_generator_train():
    train, test = tf.keras.datasets.mnist.load_data()

    img, labels = train
    img = np.asarray(img / 255.0, dtype=np.float32)
    for i in range(np.shape(img)[0]):
        yield img[i,:,:], labels[i]

def mnist_generator_test():
    train, test = tf.keras.datasets.mnist.load_data()

    img, labels = test
    img = np.asarray(img / 255.0, dtype=np.float32)
    for i in range(np.shape(img)[0]):
        yield img[i,:,:], labels[i]

class DatasetHook(tf.train.SessionRunHook):

    def __init__(self, train_iter_op, test_iter_op):
        self.train_iter_op  = train_iter_op
        self.test_iter_op = test_iter_op

        self.train_h = None
        self.test_h = None

    def after_create_session(self, session, coord):
        del coord
        self.train_h, self.test_h = session.run([self.train_iter_op, self.test_iter_op])



class MnistAE(Autoencoder):

    def __init__(self, input):
        self.embedding_size = 12
        super().__init__(input= input)

    def _define_encoder(self, current):
        current = tf.layers.dense(inputs= current, units= 500, activation= tf.nn.tanh)
        current = tf.layers.dense(inputs= current, units= 120, activation= tf.nn.tanh)

        return tf.layers.dense(inputs= current, units= self.embedding_size, activation= None)

    def _define_decoder(self, current):
        current = tf.layers.dense(inputs= current, units= 120, activation= tf.nn.tanh)
        current = tf.layers.dense(inputs= current, units= 500, activation= tf.nn.tanh)
        return tf.layers.dense(inputs= current, units= self.input.get_shape().as_list()[-1], activation= tf.sigmoid)


class MnistVAE(VariationalAutoencoder):

    def __init__(self, input):
        self.embedding_size = 2
        super().__init__(input= input)

    def _define_encoder(self, current):
        current = tf.layers.dense(inputs=current, units=50, activation=tf.nn.relu)
        current = tf.layers.dense(inputs=current, units=50, activation=tf.nn.relu)

        mu = tf.layers.dense(inputs=current, units=self.embedding_size, activation=None)
        log_sigma = tf.layers.dense(inputs=current, units=self.embedding_size, activation=None)

        return mu, log_sigma

    def _define_decoder(self, current):
        current = tf.layers.dense(inputs= current, units= 50, activation= tf.nn.relu)
        current = tf.layers.dense(inputs= current, units= 50, activation= tf.nn.relu)
        return tf.layers.dense(inputs= current, units= self.input.get_shape().as_list()[-1], activation= tf.sigmoid)


def generate_dataset(num_epochs, batch_size):

    dataset = tf.data.Dataset.from_generator(mnist_generator_train, output_types= (tf.float32, tf.int32))
    dataset = dataset.map(transform_mnist).repeat(num_epochs).batch(batch_size)

    testset = tf.data.Dataset.from_generator(mnist_generator_test, output_types= (tf.float32, tf.int32))
    testset = testset.map(transform_mnist).batch(1000)
    return dataset, testset

def transform_mnist(img, label):
    img.set_shape([28,28])
    img = tf.reshape(img,[-1])
    label = tf.one_hot(label, 10)
    return img, label

def training(mode = 'ae'):

    num_epochs = 500
    batch_size = 200

    dataset, testset = generate_dataset(num_epochs= num_epochs, batch_size= batch_size)
    string_handle = tf.placeholder(dtype= tf.string, shape= [], name= 'iterator_handle')
    iterator : tf.data.Iterator = tf.data.Iterator.from_string_handle(string_handle, dataset.output_types, dataset.output_shapes)
    next_img, next_label = iterator.get_next()

    training_iter_handle = dataset.make_one_shot_iterator().string_handle()
    test_iter = testset.make_initializable_iterator()
    test_iter_handle = test_iter.string_handle()

    ds_hook = DatasetHook(training_iter_handle, test_iter_handle)

    # define the autoencoder
    if mode == 'ae':
        ae = MnistAE(input= next_img)
    else:
        ae = MnistVAE(input= next_img)


    # visualize embeddings
    cp_dir = 'tb/mnist_ae' if mode == 'ae' else 'tb/mnist_vae'
    metadata = os.path.join(cp_dir, 'metadata.tsv')
    embedding_viz = tf.Variable(initial_value= tf.zeros([1000,12]),
                                name= 'embedding_viz')
    assign_for_viz = embedding_viz.assign(ae.latent_z)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_viz.name
    embedding.metadata_path = 'metadata.tsv'

    # optimizer
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(1e-5).minimize(ae.loss, global_step= global_step)

    # summaries
    tf.summary.scalar('Loss', ae.loss)

    if mode == 'vae':
        tf.summary.scalar('KL Loss', ae.kl_loss)
        tf.summary.scalar('Recon Loss', ae.recon_loss)


    writer = tf.summary.FileWriter(cp_dir)
    projector.visualize_embeddings(writer,config)


    # https://github.com/tensorflow/tensorflow/issues/12859
    scaffold = tf.train.Scaffold(local_init_op= tf.group(tf.local_variables_initializer(), test_iter.initializer))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    done = False

    with tf.train.MonitoredTrainingSession(hooks= [ds_hook],
                                           checkpoint_dir= cp_dir,
                                           scaffold= scaffold,
                                           config= tf.ConfigProto(gpu_options= gpu_options),
                                           save_checkpoint_steps= 100) as sess:

        while not sess.should_stop():

            # Training
            for tr in range(100):
                _, step, loss = sess.run([train_op, global_step, ae.loss], feed_dict = {string_handle : ds_hook.train_h})
                if tr % 10 == 0:
                    print('[%i] loss= %f' % (step, loss))

            # Testing
            sess.run(test_iter.initializer, feed_dict= {string_handle : ds_hook.test_h})
            _, loss, ex = sess.run([assign_for_viz, ae.loss, next_label], feed_dict= {string_handle : ds_hook.test_h})
            print('Embeddings Visualized. Test loss= %f' % loss)

            if not done:
                with open(metadata, 'w') as f:
                    for l in ex:
                        f.write('%d\n' % np.argmax(l))
                done = True

    print('done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type= str, default= 'ae')
    args = parser.parse_args()

    assert( args.mode in ['ae', 'vae'])

    training(mode = args.mode)