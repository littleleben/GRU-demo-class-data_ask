#encoding:utf-8
from collections import  Counter
import tensorflow.keras as kr
import numpy as np
import codecs
import re
import jieba


def read_file(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
    Returns:
        two list where the first is lables and the second is contents cut by jieba

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                # assert len(line.split('\t'))==2
                _,label,content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        for w in jieba.cut(blk):
                            if len(w) >= 2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels, contents

def build_vocab(filenames,vocab_dir,vocab_size=8000):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename

    """
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    print(str(all_data))
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')

def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id
        
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    categories = ['Thigh pain', 'Walking disability', 'Flatulence/wind', 'Vertigo', 'Severe pain', 'Hip pain', 'Unable to think clearly', 'Unable to walk', 'Constant pain', 'Lipitor', 'Myalgia', 'Numbness of hand', 'Pain in upper limb', 'Weakness of limb', 'Headache', 'Influenza-like illness', 'Difficulty sleeping', 'Muscle fatigue', 'Serum cholesterol raised', 'Dizziness', 'Pain', 'Rash', 'Excessive upper gastrointestinal gas', 'Pain provoked by movement', 'Stomach cramps', 'Loss of hair', 'Pain in calf', 'Charleyhorse', 'Excruciating pain', 'Loss of motivation', 'Arthritis', 'Stomach ache', 'Itching', 'Muscle atrophy', 'Depression', 'Upset stomach', 'Pain in lower limb', 'Tired', 'Asthenia', 'Lack of energy', 'Memory impairment', 'Myocardial infarction', 'Visual disturbance', 'Tremor', 'Burning sensation', 'Unable to concentrate', 'Tinnitus', 'Feeling irritable', 'Malaise', 'Cramp in calf', 'Nausea', 'Fatigue', 'Backache', 'Cramp in foot', 'Myalgia/myositis - lower leg', 'Hypersomnia', 'Impairment of balance', 'Hand pain', 'Lightheadedness', 'Myopathy', 'Knee pain', 'Poor short-term memory', 'Cramp in lower limb', 'Finding of increased blood pressure', 'ascorbic acid', 'ubidecarenone', 'Generalised aches and pains', 'Shoulder pain', 'Foot pain', 'Mentally dull', 'Chest pain', 'Insomnia', 'Arthralgia', 'Bleeding', 'Blurred vision - hazy', 'Diarrhoea', 'Muscle cramp', 'Muscle twitch', 'Neck pain', 'Serum creatinine raised', 'Pins and needles', 'Reduced libido', 'Low back pain', 'Ankle pain', 'Arthrotec', 'Muscle weakness', 'Heart disease', 'Cannot sleep at all']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id

def process_file(filename,word_to_id,cat_to_id,max_length=200):
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad

def batch_iter(x,y,batch_size=64):
    """
    Args:
        x: x_pad get from def process_file()
        y:y_pad get from def process_file()
    Yield:
        input_x,input_y by batch size

    """

    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id 
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
        
    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


def get_sequence_length(x_batch):
    """
    Args:
        x_batch:a batch of input_data
    Returns:
        sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths=[]
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths