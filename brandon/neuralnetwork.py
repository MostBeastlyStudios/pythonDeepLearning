import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# reading handwritten numbers

'''
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

^^^ Feed forward NN because it goes straight through

Compare output to intended output > cost function (cross entropy)
Optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad, etc.)

^^^ Back propogation because it resets variables based on intended results

feed forward + backprop = epoch

Training is in epoch iterations
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) #one of the nodes being on designates an answer


n_nodes_hl1 = 500 # number of nodes in hidden layer 1, 2, 3
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 # 10 classes, 0-9
batch_size = 100

X = tf.placeholder('float',[None, 784]) #matrix of pixels is 28x28, this does not need to be defined, but makes debugging easier
y = tf.placeholder('float')

def neural_network_model(data): #creates the model
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])), #randomizing weights CAN help, but may make training take longer or get worse accuracy, you're rolling the dice
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} # same

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

    layer1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3,output_layer['weights']), output_layer['biases'])

    return(output)

'''
    hidden_1_layer = {'weights':tf.Variable(1), #should set to one somehow
                        'biases':tf.Variable(1)}

    hidden_2_layer = {'weights':tf.Variable(1),
                        'biases':tf.Variable(1)}

    hidden_3_layer = {'weights':tf.Variable(1),
                        'biases':tf.Variable(1)}

    hidden_4_layer = {'weights':tf.Variable(1),
                        'biases':tf.Variable(1)}
'''

def train_neural_network(X):
    prediction = neural_network_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) #learning_rate = 0.001

    hm_epochs = 10 #how many epochs

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs): #training the data in a loop
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): #loop through data set
                epoch_X, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {X: epoch_X, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #sees if the index of the largest value of both are equal and returns bool
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))

train_neural_network(X)
