# -*- coding: utf-8 -*-
import tensorflow as tf

class TextConfig(object):

    embedding_dim = 300      #dimension of word embedding
    vocab_size =6000         #number of vocabulary
    pre_trianing=None        #use vector_char trained by word2vec

    seq_length=20          #max length of sentence
    num_classes=88         #number of labels

    num_layers= 1           #the number of layer
    hidden_dim = 128        #the number of hidden units
    attention_size = 100    #the size of attention layer


    keep_prob=0.5          #droppout
    learning_rate= 0.001    #learning rate
    lr_decay= 0.998         #learning rate decay
    grad_clip= 5.0         #gradient clipping threshold

    num_epochs=10         #epochs
    batch_size= 8         #batch_size
    print_per_batch =200   #print result

    train_filename = r'F:\ALL_model\gru\ASK_google\data\cnews.train.txt'  # train data
    test_filename = r'F:\ALL_model\gru\ASK_google\data\cnews.test.txt'  # test data
    val_filename = r'F:\ALL_model\gru\ASK_google\data\cnews.val.txt'  # validation data
    vocab_filename = r'F:\ALL_model\gru\ASK_google\data\vocab.txt'  # vocabulary
    vector_word_filename = r'F:\ALL_model\gru\ASK_google\data\vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = r'F:\ALL_model\gru\ASK_google\data\vector_word.npz'

class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequen_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn()

    def rnn(self):
        with tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                        initializer=tf.constant_initializer(self.config.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cell'):
            cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells = [cell for _ in range(self.config.num_layers)]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        with tf.name_scope('rnn'):
            # hidden一层 输入是[batch_size, seq_length, embendding_dim]
            # hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
            # 2*hidden_dim = embendding_dim + hidden_dim
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.embedding_input, sequence_length=self.seq_length,
                                          dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            # output:[batch_size, seq_length, hidden_dim]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([self.config.hidden_dim, self.config.num_classes], stddev=0.1),
                            name='w')
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
