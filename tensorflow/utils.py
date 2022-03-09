'''
Utility functions by Jack. 
'''
from os import listdir, walk
from os.path import isfile, join, exists
from collections import Counter
import json
import random
import tensorflow as tf


def get_filepathes_from_dir(file_dir, include_sub_dir=False,
                            file_format=None, shuffle=False):
    
    if include_sub_dir:
        filepathes = []
        for root, _, files in walk(file_dir, topdown=False):
            for f in files:
                filepathes.append(join(root, f))
    else:
        filepathes = [join(file_dir, f) for f in listdir(file_dir)
                      if isfile(join(file_dir, f))]
        
    if file_format:
        if not isinstance(file_format, (str, list, tuple)):
            raise TypeError("file_format must be str, list or tuple.")
        file_format = tuple(file_format) if isinstance(file_format, list) else file_format
        format_checker = lambda f: f.endswith(file_format)
        filepathes = list(filter(format_checker, filepathes))

    if shuffle:
        random.shuffle(filepathes)
    else:
        pass
        
    return filepathes


def sort_filepathes(filepathes):
    get_size = lambda x: int(x.split('/')[-2].replace('k', ''))
    get_subclass = lambda x: x.split('/')[3]
    filepathes = sorted(filepathes, key=get_size)
    filepathes = sorted(filepathes, key=get_subclass)
    return filepathes


def get_two_test_fpathes(train_path):
    t1 = train_path.replace('Training', 'Test1')
    t2 = train_path.replace('Training', 'Test2')
    return t1, t2


def load_dataset(fpath, num_row_to_skip=0):
    
    def relabel(label):
        if label == 'FALSE':
            return 0
        if label == 'TRUE':
            return 1 
        raise ValueError(f"Unknown label: {label}")
    
    def read(path):
        data = open(path)
        for _ in range(num_row_to_skip):
            next(data)
    
        for line in data:
            line = line.split('\t')
        
            yield [line[0], relabel(line[1].rstrip())]
    
    if isinstance(fpath, str):
        assert exists(fpath), f"{fpath} does not exist!"
        return list(read(fpath))
    
    elif isinstance(fpath, (list, tuple)):
        for fp in fpath:
            assert exists(fp), f"{fp} does not exist!"
        return [list(read(fp)) for fp in fpath]
    
    raise TypeError("Input fpath must be a (list) of valid filepath(es)")


def transform(dataset, encoder, num_class, shuffle=True,
              max_len=None, pad_idx=0, include_label=True):
    
    if shuffle:
        random.shuffle(dataset)
    
    if include_label:
        X = [encoder(t[0]) for t in dataset]
    else:
        X = [encoder(t) for t in dataset]
        
    if not max_len:
        max_len = max(len(x) for x in X)    
    
    for idx, text_ids in enumerate(X):
        dif = max_len - len(text_ids)
        
        if dif > 0:
            X[idx] = [pad_idx] * dif + text_ids
        else:
            X[idx] = text_ids[:max_len]         
    
    X = tf.convert_to_tensor(X, dtype=tf.int64)
    
    if include_label:
        Y = [tf.one_hot(t[-1], depth=num_class) for t in dataset]
        Y = tf.convert_to_tensor(Y, dtype=tf.int64)
        return X, Y
    
    return X


class TextVectorizer:
    '''Note that special token 'UNK' is not added here as the vocab is known.'''
    def __init__(self, tokenizer, preprocessor=None):
        if preprocessor:
            self.tokenize = lambda tx: tokenizer(preprocessor(tx))
        else:
            self.tokenize = tokenizer        
        
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.vocab_freq_count = None
    
    def _create_vocab_dicts(self, unique_tks):
        for tk in ['[PAD]']:
            if tk in unique_tks:
                unique_tks.remove(tk)
         
        unique_tks = ['[PAD]'] + unique_tks
        
        self.vocab_to_idx = {tk: i for i, tk in enumerate(unique_tks)}        
        self.idx_to_vocab = {i: v for v, i in self.vocab_to_idx.items()}
        
        print('Two vocabulary dictionaries have been built!\n' \
             + 'Please call \033[1mX.vocab_to_idx | X.idx_to_vocab\033[0m to find out more' \
             + ' where [X] stands for the name you used for this TextVectorizer class.')
        
    
    def build_vocab(self, text, top=None, random=False):
        
        if isinstance(text, str):
            tks = self.tokenize(text)

        elif isinstance(text, (list, tuple,)):
            assert all(isinstance(t, str) for t in text), f'text must be a list/tuple of str'
        
            tks = []
            for t in text:
                tks.extend(self.tokenize(t))
        else:  
            raise TypeError(f'Input text must be str/list/tuple, but {type(text)} was given')
    
        if random:
            make_vocab = lambda tks: list(set(tks))[:top]
        else:
            self.vocab_freq_count = Counter(tks).most_common()
            make_vocab = lambda tks: [tk[0] for tk in self.vocab_freq_count][:top]
        
        unique_tks =  make_vocab(tks)
        
        self._create_vocab_dicts(unique_tks)
    
    def text_encoder(self, text):
        if isinstance(text, list):
            return [self(t) for t in text]
        
        tks = self.tokenize(text)
        out = [self[tk] for tk in tks]
        return out
    
    def _save_json_file(self, dic, fpath):        
        with open(fpath, 'w') as f:
            json.dump(dic, f)
            print(f"{fpath} has been successfully saved!")    
    
    def save_vocab_as_json(self, v_to_i_fpath='vocab_to_idx.json'):
        
        fmt_conv = lambda x: x + '.json' if not x.endswith('.json') else x
        v_to_i_fpath = fmt_conv(v_to_i_fpath)
        self._save_json_file(self.vocab_to_idx, v_to_i_fpath)
        
    def load_vocab_from_json(self, v_to_i_fpath='vocab_to_idx.json', msg=False):
        
        if exists(v_to_i_fpath):
            self.vocab_to_idx = json.load(open(v_to_i_fpath))
            self.idx_to_vocab = {idx: tk for tk, idx in self.vocab_to_idx.items()}
            
            if msg:
                print(f"{v_to_i_fpath} has been successfully loaded!" \
                 + " Please call \033[1mX.vocab_to_idx\033[0m to find out more.")
                print("X.idx_to_vocab has been been successfully built from X.vocab_to_idx." \
                 + " Please call \033[1mX.idx_to_vocab\033[0m to find out more.")
                print('\nWhere [X] stands for the name you used for this TextVectorizer class.')
        else:   
            raise RuntimeError(f"{v_to_i_fpath} does not exists!")
    
    def __getitem__(self, w):
        return self.vocab_to_idx[w]
    
    def __len__(self):
        return len(self.vocab_to_idx)
        
    def __call__(self, text):
        if not self.vocab_to_idx:
            try:
                self.load_vocab_from_json()
            except:
                raise ValueError("No vocab is built or loaded.")
            
        return self.text_encoder(text)
