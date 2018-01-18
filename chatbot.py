#CHATBOT

#import libraries

import numpy as np
import tensorflow as tf
import re
import time  


#PART 1 --> DATA PRE-PROCESSING

# import the dataset

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# create a dictionary that maps each line with its ID

id2line = {};

for line in lines:  
    _line = line.split(' +++$+++ ')         ## _line means it's a temporary local variable
    if len(_line) == 5 :                    ## for loop splits the given dataset into a dictionary
        id2line[_line[0]] = _line[4]
    
# create a list for all the conversations
        
conversations_ids = []

for conversation in conversations[:-1]:                                                          ## -1 because the last row does not contain anything (Range -1)
    _convo = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")         ## -1 grabs the last element in the list 
    
    conversations_ids.append( _convo.split(","))
    
#Getting the questions and answers seperateley
    
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
    
    
#Doing a first cleaning of the text (making sure its lowercase, remove appostreophes etc.)
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"[-()\"*&^%$#/@;:<>{}+=~|.?,]", "", text)
    return text


#Cleaning the questions
    
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
   

#Cleaning the answers
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))            


#create a dictionary that maps words to its number of occurences (in order to remove not important words)
    
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
 
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1       


#create a dictionary that maps each word with a unique integer, and reduces the amounts of words 5% least frequent (threshold)
            
threshold = 20

questionswords2int = {}
wordNum = 0

for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = wordNum
        wordNum += 1
    
answerswords2int = {}
wordNum = 0

for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = wordNum
        wordNum += 1    


#adding the last tokens to these 2 dictionaries
        
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

#getting the inverse of the answerswords2int dictionary (needed for the seq2seq model)

answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
    
#add the <SOS> token to all the answers that we cleaned

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

#turn all the questions and answers into unique integers (according to how the previous dictionary was made)
#and replace all the words that were filtered out by <OUT>
    
questions_into_int = []

for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []

for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

#sorting both the questions and the answers by the length of the questions (to speed up the training/optimize it)
    
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            


#PART 2 --> BUILDING THE SEQ2SEQ MODEL
            

# creating the placeholders for the inputs and the targets (tensorflow uses place holders for tensors --> fast advanced arrays)
    
    
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input' )     #(type, dimensions of the matrix, arg name )
    targets = tf.placeholder(tf.int32, [None, None], name = 'target' )     #(type, dimensions of the matrix, arg name )
    lr = tf.placeholder(tf.float32, name = 'learn_rate' ) #parameter used for the dropout rate (to control it during back propogation)
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob' )
    
    return inputs, targets, lr, keep_prob


# preprocessing the targets (adding the <SOS> token at the front and cutting the <EOS> token at the back)
    

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>']) #getting the first space to be filled with the corresponding value for <SOS>
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])  #getting everything except for that last column
    preprocessed_targets = tf.concat([left_side, right_side], 1) #using the contact function with the left and right sides
    
    return preprocessed_targets
    

# creating the encoder RNN model --> seq2seq model  ## We are using the LSTM model instead of GRU 


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length): 
    # (the inputs like the input questions, target answers, lr keep prob etc.
    # rnn_size: number of input tensors
    # we are including a dropout reguralization to the LSTM (this is what keep_prob is used for)....so a stacked LSTM with dropout (improves learning)
    #SEQUENCE LENGTH: list of the length of questions in each batch
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
    #basic LSTM cell class in tensorflow
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)  
    #were applying dropout (dropout wrapper class in tf) --> DROPOUT: the technique of deactivating a certain percentage of neurons durin the training process 
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) 
    #encoder cell --> the multi rnn cell class --> it takes the lstm dropout as argument --> multiply the matrix by the number of layers
    
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs, #model inputs
                                                       dtype = tf.float32)
    #encoder state -- > bidirectional dynamic rnn function by the nn module by tensorflow (one of the outputs for the rnn which is a state)
    #the _, is there to make sure we're only getting the second output of the tfbidirectional function, what was there was the encoder_ouptut
    #this takes the input and builds independent fwd and bwd rnns, *** u have to make sure the input size of the fwd and bwd cell matches

    return encoder_state


# Decoding the training set
   
    
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #Getting the encoder state and using it as input
    #embedded inputs --> takes a word and turns it into a vector of real numbers in order for the decoder to beable to use it
    #variable scope --> an advanced data structure that wraps tf variables --> so its basically an object of this scope
    #used to return decoder outputs in the end
    #keep prob for lstm dropout
        
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    #we want to preprocess the training data in order to prepare it for the attention process
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    #keys --> keys to be compared with the target states
    #values --> values used to construct the context vector  --> returned by the encoder and used by decoder
    #score --> used to compare similarity
    #construct --> used to build the attention state
    
    #get the training decoder function 
    #it can only work if the atttention is properly prepared
    #the attention features is gonna be the arguments
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    #NOTE: encoder state from the previous function

    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    #apply the final dropout to the decoder output
    #use the tf lib to do this
    
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)

    ## LOOK AT RESOURCES


# Decoding the text/validation set
    
    
# new observations that won't be used for the training
# used to predict the test at the end (questions were asking the chatbot)
# val. set --> using a cross validation technique to keep seperately 10% of the training set for cross validation --> to test the predictive power on new validations --> this improves accuracy)

#this time we're using the attention_decoder_fn_inference--> once the chatbot is trained it infers answers to the questions asked (unlike the previous function)

#Copy and paste from prev. funciton
    
# NOTE: we're using this function not only for the decoding of the test but also the validation
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    # 4 new arguments added (to use new function)
    #max_length --> the longest answer
    # num_words -->total number of words of all the answers
    
    #****not decoder embeddings input its the matrix
        
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)

    ## MAIN CHANGE ##
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                             encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    #this function allows us to use the output function as input (first arg.)
    

    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
   
    
    ##decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob) 
    #don't need this --> only for training
    
    return test_predictions




# Creating the decoder RNN 
    


def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    #introduce the scope
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #again apply dropout
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1) #weights initialized
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases) 
        #we are making the fully connected layer
        #at first we have the lstm layers but now we need that fully connected layer^^^
        #this is the connection between the stacked lstms and the output
        
        #we made this function above vvv
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        # cross validation (keeps 10% seperately for cross validation)
        
        decoding_scope.reuse_variables()    #we want to reuse the variables
        
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
                                           
        
        #seq length -1 to not include the last token
        

    return training_predictions, test_predictions


# BUILDING THE SEQ2SEQ MODEL!!
    
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    #inputs --> questions from the scripts
    #answers --> answers
    #encoder/decoder embedding size --> num of dimensions of the encoder/decoder matrix

    #We are now assembling everhthing
    
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0,1))
    
    # answers num words + 1 because upper bound in sequence is excluded
    
    # now we need to get the encoder state
    # we are feeding it with the embedded input
    
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    #now we have the output
    
    #we need to get the preprocessed targets for training, the embeddings matrix and then we can apply them to the decoder rnn

    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
    # this is what the embeddings matrix is (initializing it using the random uniform function (with 0 and 1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int, 
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
    




## TRAINING THE SEQ2SEQ MODEL ####
    
## Setting the Hyperparameters
    
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01 ##this is tricky to set (learns either too slow or too fast)
learning_rate_decay = 0.9 ## this is usually a common value --> either 0.9 or 1
min_learning_rate = 0.0001
keep_probability = 0.5

#defining a session (in tensorflow)
#you have to reset the graph when opening up a session

tf.reset_default_graph()
session = tf.InteractiveSession()

#Load the model inputs (from the first function when starting to build the model)
inputs, targets, lr, keep_prob = model_inputs()

#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#getting the shape fromt the inputs tensor
input_shape = tf.shape(inputs)

#getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),
                                                                  targets,
                                                                  keep_prob,
                                                                  batch_size,
                                                                  sequence_length,
                                                                  len(answerswords2int),
                                                                  len(questionswords2int),
                                                                  encoding_embedding_size,
                                                                  decoding_embedding_size,
                                                                  rnn_size,
                                                                  num_layers,
                                                                  questionswords2int)
#reverse used to reshape the inputs to fit the model


#setting up the loss error, optimizer, and gradient clipping
#gradient clipping caps the gradient in the graph at some minimum and maximum values to prevent issues

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
                                                  ##tensor of weights initialized to 1
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]                                             
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    
    
# padding the sequences with the <PAD> function
# Question: ['who', 'are', 'you']
# Answer: [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
# <PAD> tokens are added at the end in order to have the same length for both the questions and the answers
    
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

#Splitting the data into batches of questions and answers

def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int)) #need to use numpi array to transfer to tensor flow
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield  padded_questions_in_batch, padded_answers_in_batch ###double check yield vs. return
        
#Splitting questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15) ## 15 percent of the sorted clean questions

training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]

validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


#####  TRAINING!!!!! #######
        
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())

for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions,training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:  ## every 100 batches it prints this
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch, 
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error // batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
            
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            
            total_validation_loss_error = 0
            starting_time = time.time()
            
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions,validation_answers, batch_size)):
                
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: 1})
                total_training_loss_error += batch_validation_loss_error
                
            ending_time = time.time()
            batch_time = ending_time - starting_time
            
            average_validation_loss_error = total_validation_loss_error / len(validation_questions) / batch_size
            
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate  *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
                
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break 

    if early_stopping_check == early_stopping_stop:
        print("My appologies, I cannot speak better anymore. This is the best I can do.")
        break
print("GAME OVER")
































    