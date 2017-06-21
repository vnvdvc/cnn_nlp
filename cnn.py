import numpy as np
import tensorflow as tf
import os,time,errno
from vocab_utils import *
from xavier_initializer import xavier_weight_init
from datetime import datetime




class Config(object):
# Holds data information and model hyperparameters
    
    embed_size = 25
    batch_size = 32
    hidden_size = 100 
    label_size = 3
    sentence_length = 140
    filters = [1,2]
    num_of_filters = 10  #num of_filters of each size
    max_epoch = 100
    early_stopping = 2
    dropout = 0.5
    lr = 0.001
    l2_conv = 0.001
    l2_softmax = 0.001
    activation = "tanh"   #tanh or relu
    activation_hidden = "tanh"
    real_analysis = None

class Model(object):

    def load_data(self,debug=False):
    # load wv, word_to_index, index_to_word
        self.wv,self.word_to_index,self.index_to_word = load_wv(self.config.embed_size)

    # load dataset in index_form
        train_data = []
        train_labels = []
        train_gates = []
        zero_pad = len(self.index_to_word)
        for sentence, attitude in get_dataset():
            train_data.append(get_indexForm(sentence,self.word_to_index))
            train_gates.append([float(x!=zero_pad) for x in train_data[-1]])
            train_labels.append(get_polarity(attitude))
        self.train_data = np.array(train_data)
        self.train_labels = np.array(train_labels)
        self.train_gates = np.array(train_gates)
        val_data = []
        val_labels = []
        val_gates = []
        for sentence, attitude in get_dataset("val"):
            val_data.append(get_indexForm(sentence,self.word_to_index))
            val_labels.append(get_polarity(attitude))
            val_gates.append([float(x!=zero_pad) for x in val_data[-1]])
        self.val_data = np.array(val_data)
        self.val_labels = np.array(val_labels)
        self.val_gates = np.array(val_gates)
        test_data = []
        test_labels = []
        test_gates = []
        for sentence, attitude in get_dataset("test"):
            test_data.append(get_indexForm(sentence,self.word_to_index))
            test_gates.append([float(x!=zero_pad) for x in test_data[-1]])
            test_labels.append(get_polarity(attitude))
        self.test_data = np.array(test_data)
        self.test_labels = np.array(test_labels)
        self.test_gates = np.array(test_gates)
        
        if debug:
            self.train_data = self.train_data[:1024,:]
            self.train_labels = self.train_labels[:1024,:]
            self.train_gates = self.train_gates[:1024,:]
            self.val_data = self.val_data[:1024,:]
            self.val_labels = self.val_labels[:1024,:]
            self.val_gates = self.val_gates[:1024,:]
            self.test_data = self.test_data[:1024,:]
            self.test_labels = self.test_labels[:1024,:]
            self.test_gates = self.test_gates[:1024,:]
	
    def load_real_data(self,datafile):
	    self.wv,self.word_to_index,self.index_to_word = load_wv(self.config.embed_size)

    # load real dataset in index_form, labels are all zeros
            real_data = []
            real_gates = []
            real_labels = []
            zero_pad = len(self.index_to_word)
            for sentence, attitude in get_dataset(datafile):
                real_data.append(get_indexForm(sentence,self.word_to_index))
                real_gates.append([float(x!=zero_pad) for x in real_data[-1]])
            	real_labels.append(get_polarity(attitude))
            self.real_data = np.array(real_data)
            self.real_gates = np.array(real_gates)
	    self.real_labels = np.array(real_labels)


    def add_placeholders(self):
        '''
        self.x_placeholder: shape (batch_size,sentence_length) 
        self.label_placeholder: shape (batch_size, label_size)
        self.dropout_placeholder: scalar
        '''
        self.x_placeholder = tf.placeholder(tf.int32,shape=(None,self.config.sentence_length),name="x")
        self.gate_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.sentence_length),name="gate")
        self.label_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.label_size),name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_embedding(self):
        '''
        self.wv: shape (size_of_vocabulary, embed_size)
        return:
        sentences: shape (batch_size,sentence_size,embed_size)
        '''
        with tf.device('/cpu:0'):
            sentences = tf.nn.embedding_lookup(tf.constant(self.wv),self.x_placeholder,name="embedding")
       
        self.embedding_expanded = tf.cast(tf.expand_dims(sentences,-1),tf.float32) 
        #The input of conv2d requires a 4D tensor
        

    def create_feed_dict(self,x_batch,labels_batch,gate_batch,dropout=None):
        '''
        input_x is an array of indexes representing words of a sentence with shape (batch_size, None)
        '''
        feed_dict = { 
                    self.x_placeholder: x_batch,
                    self.label_placeholder: labels_batch,
                    self.gate_placeholder: gate_batch
                }
        if dropout is None:
            feed_dict[self.dropout_placeholder] = self.config.dropout
        else:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict


    def filter_max_pooling(self,activation="tanh"):
        '''
        construct hidden states c: shape (None,embed_size) 
        coefficients:
        Wc: shape (filter*embed_size,embed_size)
        bc: shape (None,embed_size)
        '''
        features = []
        for filter in self.config.filters:
            name = "conv_maxpool_{}".format(filter)
            with tf.variable_scope(name),tf.name_scope(name):
            #Convolution Layer
                filter_shape = [filter,self.config.embed_size,1,self.config.num_of_filters]
                Wc = tf.get_variable("W",shape=filter_shape,initializer=xavier_weight_init())
                bc = tf.get_variable("b",shape=[self.config.num_of_filters],initializer=xavier_weight_init())
                conv = tf.nn.conv2d(
                            self.embedding_expanded,
                            Wc,
                            strides=[1,1,1,1],
                            padding = "VALID",
                            name = "raw_conv"
                        )
                filter_gate_shape = [filter,1,1,self.config.num_of_filters]
                Wg = tf.constant(1.0/filter,dtype=tf.float32,shape=filter_gate_shape)
                gates_expanded = tf.expand_dims(tf.expand_dims(self.gate_placeholder,-1),-1)
                conv_gates = tf.nn.conv2d(
                            gates_expanded,
                            Wg,
                            strides=[1,1,1,1],
                            padding = "VALID",
                            name = "conv_gates"
                        )
                gated_conv = tf.multiply(tf.nn.bias_add(conv,bc),conv_gates)
                if activation == "relu":
                    c = tf.nn.relu(gated_conv,name="relu")
                else:
                    c = tf.tanh(gated_conv,name="tanh")

                #max-pooling
                pooled = tf.nn.max_pool(
                            c,
                            ksize=[1,self.config.sentence_length-filter+1,1,1],
                            strides=[1,1,1,1],
                            padding="VALID",
                            name="pool"
                        )
                features.append(pooled)
                if self.config.l2_conv:
                    reg = self.config.l2_conv/2*tf.reduce_sum(tf.multiply(Wc,Wc))
                    tf.add_to_collection('total_loss',reg)



        #Combine all features
        total_num_filters = self.config.num_of_filters*len(self.config.filters)
        all_features = tf.concat(features,3)
        return tf.reshape(all_features,[-1,total_num_filters])


    def add_hidden_layer(self,features,activation):
        total_num_filters = self.config.num_of_filters*len(self.config.filters)
        with tf.variable_scope("hidden_layer"),tf.name_scope("hidden_layer"):
            W = tf.get_variable("W",shape=[total_num_filters,self.config.hidden_size],initializer=xavier_weight_init())
            b = tf.get_variable("b",shape=[self.config.hidden_size],initializer=xavier_weight_init())
            linear_h = tf.nn.xw_plus_b(features,W,b)
            if self.config.l2_softmax:
                reg = self.config.l2_softmax/2*tf.reduce_sum(tf.multiply(W,W))
                tf.add_to_collection('total_loss',reg)
            if activation == "sigmoid":
                h = tf.sigmoid(linear_h,name="hidden_layer")
            elif activation == "relu":
                h = tf.nn.relu(linear_h,name="hidden_layer")
            elif activation == "tanh":
                h = tf.tanh(linear_h,name="hidden_layer")
            else:
                h = tf.nn.softmax(linear_h,name="hidden_layer")


        return h

    """
    if config.activation_hidden is None, then the model has only one cnn layer, else it will have another fully connected hidden layer
    """
    def add_scores(self,h):
        with tf.variable_scope("output"):
            if self.config.activation_hidden is None:
                last_size = self.config.num_of_filters*len(self.config.filters)
            else:
                last_size = self.config.hidden_size


            W = tf.get_variable("W",shape=[last_size,self.config.label_size],initializer=xavier_weight_init())
            b = tf.get_variable("b",shape=[self.config.label_size],initializer=xavier_weight_init())
            scores = tf.nn.xw_plus_b(h,W,b)
            if self.config.l2_softmax:
                reg = self.config.l2_softmax/2*tf.reduce_sum(tf.multiply(W,W))
                tf.add_to_collection('total_loss',reg)
            
        
        return scores

    def add_loss(self,scores):
        if self.dropout_placeholder is not None:
            y = tf.nn.dropout(scores,self.dropout_placeholder)
        with tf.name_scope("loss"):
             loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=self.label_placeholder))
             tf.add_to_collection('total_loss',loss)
        
        return tf.add_n(tf.get_collection('total_loss')) 
        

    def add_training_op(self,loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        with tf.name_scope("train"):
            train_op = optimizer.minimize(loss)
            return train_op

    def add_predictions(self,scores):
        return tf.argmax(scores,1,name="predictions")

    def add_predictions_3d(self,scores):
        return tf.nn.softmax(scores,dim=-1)

    def __init__(self,config):
        self.config = config
        self.LOGDIR = "./log/{}d-{}/".format(config.embed_size,config.activation)
        if config.activition_hidden is None:
            self.LOGDIR = self.LOGDIR + "1layer-{},{}/".format(config.num_of_filters,config.filters)
            self.LOGDIR = self.LOGDIR + "lr={},l2_conv={}/".format(config.lr,config.l2_conv)
        else:
            self.LOGDIR = self.LOGDIR + "2layers-{}d{}-{},{}/".format(config.hidden_size,config.activation_hidden,config.num_of_filters,config.filters)
            self.LOGDIR = self.LOGDIR + "lr={},l2_conv={},l2_soft={}/".format(config.lr,config.l2_conv,config.l2_softmax)
	      if config.real_analysis is None:
            self.load_data()
	      else:
	          self.load_real_data(config.real_analysis)
        self.add_placeholders()
        self.add_embedding()
        features = self.filter_max_pooling(self.config.activation)
        if self.config.activation_hidden is None:
            h = features
        else:
            h = self.add_hidden_layer(features,self.config.activation_hidden)
        scores = self.add_scores(h)
        self.loss = self.add_loss(scores)
        self.predictions = self.add_predictions(scores)
        self.predictions_3d = self.add_predictions_3d(scores)
        self.train_op = self.add_training_op(self.loss)
        self.epoch = 0

    def run_epoch(self,session,shuffle=True):
        config = self.config
        dp = self.config.dropout
        total_losses = []
        total_correct_examples = 0
        total_processed_examples = 0
        for x,y,g in data_iterator(self.train_data,self.train_labels,self.train_gates,config.batch_size,shuffle=shuffle):
            feed = self.create_feed_dict(x,y,g,dp)
            loss, predictions, _ = session.run(
                [self.loss,self.predictions,self.train_op],feed_dict=feed)
            correct_predictions = np.sum(np.equal(np.argmax(y,axis=-1),predictions).astype(int))
            total_correct_examples += correct_predictions
            total_processed_examples += len(x)
            accuracy = float(correct_predictions)/len(x)
            total_losses.append(loss)
            accuracy = total_correct_examples/float(total_processed_examples)


        loss = np.mean(total_losses)

        return loss, accuracy

    def predict(self,session,test=False):
        config = self.config
        dp = 1.0 
        total_losses = []
        confusion = np.zeros((self.config.label_size,self.config.label_size),dtype=np.int32)
        checkpt = os.path.join(self.LOGDIR,"checkpt.tsv")
        if test:
            data = self.test_data
            labels = self.test_labels
            gates = self.test_gates
        else:
            data = self.val_data
            labels = self.val_labels
            gates = self.val_gates
        results = None
        for x,y,g in data_iterator(data,labels,gates,config.batch_size):
            feed = self.create_feed_dict(x,y,g,dp)
            loss, predictions, predictions_3d = session.run([self.loss,self.predictions,self.predictions_3d],feed_dict=feed)
            total_losses.append(loss)
            answers = np.argmax(y,axis=1)
            """
            if test:
                with open(checkpt,"a+") as ckpt_file:
                    for pred3d in predictions_3d:
                        ckpt_file.write(str(pred3d[0])+'\t'+str(pred3d[1])+'\t'+str(pred3d[2])+'\n')
            """
            for i in xrange(len(answers)):
                correct_label = answers[i]
                pred_label = predictions[i]
                confusion[correct_label,pred_label] += 1

            if results is None:
              results = predictions_3d
            else:
              results = np.concatenate(results,predictions_3d)

        # projections are predicted results projected onto the axis corresponding to the answers
        if test is False:
          projections = np.sum(labels*results,axis=-1)
          proj_str = ','.join([str(item) for item in projections])
          proj_file = self.LOGDIR + "projs.tmp"

          with open(proj_file,"a+") as out:
            out.write(proj_str+"\n")
          

        accuracy = np.sum(np.diag(confusion))/float(np.sum(confusion))
        loss = np.mean(total_losses)

        return loss,accuracy,confusion 

def print_confusion(confusion):
    num_to_tag = dict()
    num_to_tag[0] = "positive"
    num_to_tag[1] = "neutral"
    num_to_tag[2] = "negative"
    total_guessed_tags = confusion.sum(axis=0)
    total_true_tags = confusion.sum(axis=1)
    print confusion
    for i,tag in sorted(num_to_tag.items()):
        prec = confusion[i,i] / float(total_guessed_tags[i])
        recall = confusion[i,i] / float(total_true_tags[i]) 
        print 'Tag: {}  - P {:2.4f} / R {:2.4f}'.format(tag,prec,recall)

def save_test_results(config,results_file,loss,confusion,loss_and_acc):
    num_to_tag = dict()
    num_to_tag[0] = "positive"
    num_to_tag[1] = "neutral"
    num_to_tag[2] = "negative"
    total_guessed_tags = confusion.sum(axis=0)
    total_true_tags = confusion.sum(axis=1)
    now = str(datetime.now()) 
    with open(results_file,"a+") as out:
        out.write(now + "\n")
        out.write("Results of twitter datasets using cnn:\n")
      	out.write("{}d-{} with batch {}\n".format(config.embed_size,config.activation,config.batch_size))
        if config.activation_hidden is None:
          out.write("1layer-{}d,{} filters\n".format(config.num_of_filters,",".join([str(num) for num in config.filters])))
          out.write("lr={},l2_conv={},l2_soft={}\n".format(config.lr,config.l2_conv,config.l2_softmax))
        else:
          out.write("2layers-hidden_{},{}d-{}d,{} filters\n".format(config.activation_hidden,config.hidden_size,config.num_of_filters,",".join([str(num) for num in config.filters])))
          out.write("lr={},l2_conv={},l2_soft={}\n".format(config.lr,config.l2_conv,config.l2_softmax))
      	out.write("dropout: {}\n".format(config.dropout))

        out.write("Losses of train and val datasets:\n")
        out.write(','.join([str(item) for item in loss_and_acc[0]])+'\n')
        out.write(','.join([str(item) for item in loss_and_acc[1]])+'\n')
        out.write("Accuracies of train and val datasets:\n")
        out.write(','.join([str(item) for item in loss_and_acc[2]])+'\n')
        out.write(','.join([str(item) for item in loss_and_acc[3]])+'\n')

        out.write("test dataset results:\n")
        out.write("true//guess, {},{},{}\n".format(num_to_tag[0],num_to_tag[1],num_to_tag[2]))
        out.write("postive, {},{},{}\n".format(confusion[0,0],confusion[0,1],confusion[0,2]))
        out.write("neutral, {},{},{}\n".format(confusion[1,0],confusion[1,1],confusion[1,2]))
        out.write("negative, {},{},{}\n".format(confusion[2,0],confusion[2,1],confusion[2,2]))
        for i,tag in sorted(num_to_tag.items()):
            prec = confusion[i,i] / float(total_guessed_tags[i])
            recall = confusion[i,i] / float(total_true_tags[i]) 
            out.write('Tag: {}  - P {:2.4f} / R {:2.4f}\n'.format(tag,prec,recall)) 
        accuracy = np.sum(np.diag(confusion))/float(np.sum(confusion))
        out.write('Accuracy of Test: {}\n\n'.format(accuracy))

def save_projections(logdir):
    projs_tmp = logdir + "projs.tmp"
    arr = []
    inp = open(projs_tmp,"r")
    temp = inp.readline()
    while temp:
        if temp[-1] == '\n':
            temp = temp[:-1]
        arr.append([float(num) for num in temp.split(',')])
        temp = inp.readline()
    inv_arr = np.array(arr).T
    
    projs_file = logdir + "projs.txt"
    if os.path.isfile(projs_file):
       temp = projs_file[:-4] + "0.txt"
       while os.path.isfile(temp):
         temp = temp[:-4] + "0.txt"
       projs_file = temp

    for line in inv_arr:
        with open(projs_file,"a+") as out:
            out.write(','.join([str(item) for item in line])+'\n')

    os.remove(projs_tmp)

    

def directory_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
                raise

def run_cnn(config,results_file):
    with tf.Graph().as_default():
         model = Model(config)
         init = tf.global_variables_initializer()
         saver = tf.train.Saver()

         with tf.Session() as session:

             best_val_loss = 10.0
             best_val_epoch = 0
             session.run(init)
             directory_exists(model.LOGDIR)
             val_losses =[]
             val_accs =[]
             train_losses = []
             train_accs =[]
             for epoch in xrange(config.max_epoch):
                  print "Epoch: {}".format(epoch)
                  start = time.time()
                  model.epoch = epoch
                  if model.epoch == 0:
                     projs_tmp = model.LOGDIR + "projs.tmp"
                     if os.path.isfile(projs_tmp):
                        print "Warning: existing projs.tmp file, deleted.\n"
                        os.remove(projs_tmp)

                  train_loss, train_acc  = model.run_epoch(session,shuffle=True) 
                  val_loss,val_acc,val_confusion = model.predict(session)

                  train_losses.append(train_loss)
                  train_accs.append(train_acc)
                  val_losses.append(val_loss)
                  val_accs.append(val_acc)

                  print "training loss: {}, training accuracy: {}".format(train_loss,train_acc)
                  print "validation loss: {}, confusion:\n".format(val_loss)
                  print_confusion(val_confusion)

                  if val_loss < best_val_loss:
                      best_val_loss = val_loss
                      best_val_epoch = epoch

                      saver.save(session,"./weights/cnn-weights")

                  if epoch - best_val_epoch > config.early_stopping:
                      break

                  print "Total time: {}".format(time.time()-start)
             
             save_projections(model.LOGDIR)
             loss_and_acc = np.concatenate((train_losses,val_losses,train_accs,val_accs),axis=0)
              
             saver.restore(session,"./weights/cnn-weights")
             print "Test:"
             test_loss,test_acc,test_confusion = model.predict(session,test=True)
             print "test loss: {}, confusion:\n".format(test_loss)
             print_confusion(test_confusion)
             save_test_results(config,results_file,test_loss,test_confusion,loss_and_acc)

def param_opt(results_file):
    
    if os.path.isfile(results_file):
       temp = results_file[:-4] + '0.txt'
       while os.path.isfile(temp):
         temp = temp[:-4] + '0.txt'
       results_file = temp
    
    lr_list = [0.001]
    l2_conv_list = [1E-6]
    l2_softmax_list = [1E-6]
    hidden_size_list = [100]
    num_of_filters_list = [100]
    dropout_list = [0.5]
    filters_list = [[2,3,6,7,10,13],[3,6,7,10],[3,6,7,13,18],[6,7,10]]
    activation_hidden_list = ["tanh"]
    config = Config()
    for lr in lr_list:
        config.lr = lr
        for l2_conv in l2_conv_list:
            config.l2_conv = l2_conv
            for l2_softmax in l2_softmax_list:
                config.l2_softmax = l2_softmax
                for num_of_filters in num_of_filters_list:
                    config.num_of_filters = num_of_filters
                    for dropout in dropout_list:
                        config.dropout = dropout
                        for filters in filters_list:
                            config.filters = filters
                            for hidden_size in hidden_size_list:
                                config.hidden_size = hidden_size
                                for activation_hidden in activation_hidden_list:
                                    config.activation_hidden = activation_hidden
                                    run_cnn(config,results_file)

def weight_function(pred):
    if pred == 0:
        return 1.0
    elif pred == 1:
        return 0.0
    elif pred == 2:
        return -1.0

def weighted_bias_function(config):
    with tf.Graph().as_default():
         config.dropout = 1.0
         model = Model(config)
         init = tf.global_variables_initializer()
         saver = tf.train.Saver()

         weighted_bias = 0
         num_of_sentences = 0
         with tf.Session() as session:
             session.run(init)
             saver.restore(session,"./cnn-weights")
             data = self.real_data
             labels = self.real_labels
             gates = self.real_gates
             
             for x,y,g in data_iterator(data,labels,gates,config.batch_size,shuffle=False,retrain=False,real=True):
                 num_of_sentences += config.batch_size
                 feed = self.create_feed_dict(x,y,g)
                 predictions = session.run(self.predictions,feed_dict=feed)
                 for pred in predictions:
                     weighted_bias += weight_function(pred)
             
             return weighted_bias/num_of_sentences

def a_lot_of_samples(config,output='./total_biases.txt',input_dir='./real_data/'):
    files = os.listdir(input_dir)
    print "Analyzing: "
    for filename in files:
        if filename[-4:].lower() == '.txt':
            print filename
            config.real_analysis = filename
            bias = weighted_bias_function(config)
            print "bias:    {}".format(bias)
            with open(output,"w+") as out:
                 out.write(filename + '  ' + bias + '\n')
        
    
		



if __name__=="__main__":
  now = str(datetime.now())
  results_file = "results_" + now[5:10]+"_0.txt" 
  param(results_file)





