## Submission.py for COMP6714-Project2
#Written by Hao Chen 
#z5102446
###################################################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import math
import random
import zipfile
import spacy
from spacy.en import English
import numpy as np
import tensorflow as tf
import re
import collections
from tempfile import gettempdir
import zipfile
import gensim
from sklearn.manifold import TSNE

data_index = 0

#1. Read the row data using spaCy
#2. Process the data and write out the processed file in PWD
def process_data(input_data):
    parser = English()
    stop_words = [
 'a','and','that', 'he', 'i', 'is', 'are', 'her', 'us', 'with', 'its', 'very', 'dr','mr','mrs',
 'has', 'have', '£', 'of', 'for','in', 'on','at','you', 'm', 'him', "n't", 'than',
 'my', 'the', "'s", "no", "to", "for", 'UNK', 'NUM', 'but', 'so', 'or', 'our'
 'and', 'it', 'they', 'them', 'how', 'not', 'too', 'also', 'what', 'when','we'
 'me', 'over', 'if', 'does', 'did', 'do', 'have', 'had', 'by', 'about', 'be', 'via',
 '$', '£', 'as','only','here','be', 'all','were', 'about', 'such', 'should', 'an', 'who', 'will',
 'into', 'next', 'was', 'from', 'out', 'could', 'their', 'this', 'other', 'since',  
 'while', 'however', 'she', 'up', 'may', 'would', 'any', 'behind', 'been', 'ever', 'some', 'after', '\''
 ]
    try:
        if not os.path.exists(input_data):
            raise IOError
            
            
        else:
            with zipfile.ZipFile(input_data) as f:
                output_file = 'Processed data.txt'
                if os.path.exists(output_file):
                    print("File already exists!")
                    with open(output_file) as file:
                        return file.name
                else:
#                    with zipfile.ZipFile(input) as file:
#                        data = ' '.join([file.read(fileName).decode() for fileName in file.namelist()])
#                    parsedData = parser(data)
#                    with open(output_file, 'w', encoding='utf-8') as f:
#                        for token in parsedData:
#                            if token.pos_ not in ['PUNCT', 'SPACE', 'NUM']:
#                                if not token.is_stop:
#                                    f.write(token.lemma_.lower() + " ")
#                    return f.name
                    with open(output_file, 'w') as file:
                        for name in f.namelist():
                            doc = str(f.read(name), encoding ="utf-8")
                            # doc.replace("'s","")
                            parsedData = parser(doc)
                            # Let's look at the part of speech tags of the first sentence
                            for span in parsedData.sents:
                                sent = [parsedData[i] for i in range(span.start, span.end)]
                                for token in sent:
                                    if token.pos_ == "PUNCT" or token.pos_ =="SPACE" or token.pos_ =="NUM":
                                        continue
                                    else:
                                        if str(token.orth_).lower() not in stop_words:
                                            file.write(str(token.orth_).lower())
                                            file.write(' ')
                    return file.name
    except IOError:
        print("Cannot find the file")



def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
#        if word in Target_words:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_batch(batch_size, num_samples, skip_window, data):

    global data_index
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
#    print("----length---")
#    print(len(data))
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])  # initial buffer content = first sliding window

#    print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))

    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words)  # now we obtain a random list of context words
        for j in range(num_samples):  # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word]

        # slide the window to the next position
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])  # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1

#        print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))

    # end-of-for
    data_index = (data_index + len(data) - span) % len(data)  # move data_index back by `span`
    return batch, labels



#1. Train embeddings using processed data from Part2-a
#2. Write out trained embeddings in the file "adjective_embeddings.txt"
#3. Store each float in the "adjective_embeddings.txt" up to 6 decimal places.
def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):

    with open(data_file) as f:
        f_list = f.read().split()
    vocabulary_size = 5000
    batch_size = 64  
    embedding_size = embedding_dim  
    skip_window = 3 
    num_samples = 4  
    learning_rate=0.002 # How many times to reuse an input to generate a label.
    num_sampled = 64 # Sample size for negative examples.
    
    global data_index 
    
    
    logs_path = './log/'
    
    
    data, count, dictionary, reverse_dictionary = build_dataset(f_list, vocabulary_size)

    # Specification of test Sample:
    sample_size = 20  # Random sample of words to evaluate similarity.
    sample_window = 100  # Only pick samples in the head of the distribution.
    
    sample_examples = np.random.choice(sample_window, sample_size, replace=False)  # Randomly pick a sample of size 16
    
    graph = tf.Graph()
    
    #Constructing the graph...
    with graph.as_default():
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    
            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
                                                                 labels=train_labels, inputs=embed,
                                                                 num_sampled=num_sampled, num_classes=vocabulary_size))
    
            # Construct the Gradient Descent optimizer using a learning rate of 0.01.
            with tf.name_scope('Gradient_Descent'):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
    
            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
    
            # Add variable initializer.
            init = tf.global_variables_initializer()
    
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()
    
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
        print('Initializing the model')
    
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window, data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
    
            summary_writer.add_summary(summary, step)
            average_loss += loss_val
    
            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000
                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0
    
            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                nlp = spacy.load('en')
                sim = similarity.eval()  #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    
                    doc = nlp(sample_word)
                    for token in doc:
                        if token.pos_ == "ADJ":
                            top_k = 10  # Look for top-10 neighbours for words in sample set.
                            
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to %s:' % sample_word
                            for k in range(top_k):
                                close_word = reverse_dictionary[nearest[k]]
                                log_str = '%s %s,' % (log_str, close_word)
                            print(log_str)
                print()
        
    
        final_embeddings = normalized_embeddings.eval()
        
        with open(embeddings_file_name, 'w', encoding="utf-8") as file:
            file.write(str(len(dictionary)))
            file.write(' ')
            file.write(str(embedding_dim))
            file.write('\n')
            i = 0
            for key in dictionary:
                file.write(key)
                file.write(' ')
                for j in final_embeddings[i]:
                    file.write('{:.6f}'.format(j))
#                    file.write(str(round(int(j),6)))
                    file.write(' ')
                file.write('\n')
                i += 1


#1. Read the model file useing python library
#2. Read the input _adjective
#3. Return top_k most similar words
def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    result = [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k)]
    return result


