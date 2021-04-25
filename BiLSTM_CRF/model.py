import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import sklearn
import tensorflow as tf
from keras import Sequential
from keras import backend as K
from keras.layers import Embedding, Bidirectional, LSTM
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa
from tensorflow_addons import layers
from tensorflow.python.util import compat
from sklearn.metrics import recall_score, precision_score,f1_score


parser = argparse.ArgumentParser(description='BiLSTM-CRF')
parser.add_argument('--train_data', type=str, default='data/msra_train.data', help='train data source')
parser.add_argument('--test_data', type=str, default='data/msra_test.data', help='test data source')
parser.add_argument('--char2Id_file', type=str, default='data/vocab.json', help='char2Id file')
parser.add_argument('--gen_vocab', type=bool, default=True,
                    help='if true, gen vocab file. else load from exists file.')
parser.add_argument('--batch_size', type=int, default=128, help='sample of each minibatch')
parser.add_argument('--embedding_size', type=int, default=50, help='length for embedding array')
parser.add_argument('--epoch', type=int, default=30, help='#epoch of training')
parser.add_argument('--mode', type=str, default='train', help='train/test/predict/retrain')
parser.add_argument('--input_model_dir', type=str, default="model/29.h5", help='model for test or demo')
parser.add_argument('--output_model_dir', type=str, default='model', help='model to save')
args = parser.parse_args()

# all tags
CHUNK_TAGS = ["O",
              "B-PER", "I-PER",
              "B-ORG", "I-ORG",
              "B-LOC", "I-LOC"
              ]

#对训练文件和测试文件进行编码
class Word2Id:
    def __init__(self, file):
        self.file = file

#使用新的训练文件需要生成新的
    def gen_save(self):
        data_file = [args.train_data, args.test_data]
        all_char = []
        for f in data_file:
            file = open(f, "rb")
            data = file.read().decode("utf-8")
            data = data.split("\n\n")
            data = [token.split("\n") for token in data]
            data = [[j.split("\t") for j in i] for i in data]
            data.pop()
            all_char.extend([char[0] if char else 'unk' for sen in data for char in sen])
        chars = set(all_char)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        with open(self.file, "wb") as f:
            f.write(json.dumps(word2id, ensure_ascii=False).encode('utf-8'))
#其余情况加载
    def load(self):
        return json.load(open(self.file, 'rb'))


#处理数据集
class DataSet:
    def __init__(self, data_path, labels):
        with open(data_path, "rb") as f:
            #使用utf-8解码
            self.data = f.read().decode("utf-8")
        self.process_data = self.process_data()
        self.labels = labels

    def process_data(self):
        # 读取样本并分割
        train_data = self.data.split("\n\n")
        #print(train_data)
        #print("#########################")
        train_data = [token.split("\n") for token in train_data]
        #print(train_data)
        train_data = [[j.split("\t") for j in i] for i in train_data]
        #print(train_data)
        train_data.pop()
        return train_data



    def generate_data(self, vocab, maxlen):
        #获取句子
        char_data_sen = [[token[0] for token in i] for i in self.process_data]
        #获取标签
        label_sen = [[token[1] for token in i] for i in self.process_data]
        # 对样本进行编码
        sen2id = [[vocab.get(char, 0) for char in sen] for sen in char_data_sen]
        # 对样本中的标签进行编码
        label2id = {label: id_ for id_, label in enumerate(self.labels)}

        lab_sen2id = [[label2id.get(lab, 0) for lab in sen] for sen in label_sen]
        # padding
        sen_pad = pad_sequences(sen2id, maxlen)
        lab_pad = pad_sequences(lab_sen2id, maxlen, value=-1)
        lab_pad = np.expand_dims(lab_pad, 2)
        return sen_pad, lab_pad

#存放训练的模型
#通过skearn计算f1
class CallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        # self.validation_data = valid_data
        self.log_txt = '%s/logs.txt' % 'model'
        if os.path.exists(self.log_txt):
            os.rename(self.log_txt, '%s_%s' % (self.log_txt, int(time.time())))
        with open(self.log_txt, 'a+') as f:
            f.write(' '.join(sys.argv))
    # 每个epoch save一次模型
    def on_epoch_end(self, epoch, logs=None):
        #保存模型
        save_dir = '%s' % args.output_model_dir
        self.export_saved_model(save_dir, epoch)
        save_dir = '%s/%s.h5' % (save_dir, epoch)
        self.model.save(save_dir)
        print("model saved to to dir %s. %s" % (save_dir, logs))


        # logs = logs or {}
        # val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        # val_targ = self.validation_data[1]
        # if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
        #     val_targ = np.argmax(val_targ, -1)
        # _val_f1 = f1_score(val_targ, val_predict, average='macro')
        # _val_recall = recall_score(val_targ, val_predict, average='macro')
        # _val_precision = precision_score(val_targ, val_predict, average='macro')

        #
        # logs['val_f1'] = _val_f1
        # logs['val_recall'] = _val_recall
        # logs['val_precision'] = _val_precision
        # print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))

        with open(self.log_txt, 'a+') as f:
            if epoch == 0:
                self.model.summary(print_fn=lambda *args: f.write(' '.join(args) + '\n'))
            f.write('[%d] %s: %s\n' % (epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), logs))

    # 导出模型
    def export_saved_model(self, saved_dir, epoch):
        model_version = epoch
        model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'input': self.model.input}, outputs={'output': self.model.output})
        export_path = os.path.join(compat.as_bytes(saved_dir), compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature
            })
        builder.save()

#搭建模型
class Ner:
    def __init__(self, vocab, labels_category, Embedding_dim=200):
        self.Embedding_dim = Embedding_dim
        self.vocab = vocab
        self.labels_category = labels_category
        self.model = self.build_model()

    # 构建模型
    def build_model(self):
        model = Sequential()
        # embedding 层
        model.add(Embedding(len(self.vocab), self.Embedding_dim, mask_zero=True))  # Random embedding
        # bi-lstm层
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        # crf 层
        crf = tfa.layers.CRF(len(self.labels_category), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    # 训练方法
    def train(self, train_data, train_label, EPOCHS):
        self.model.fit(train_data, train_label, batch_size=args.batch_size,callbacks=[CallBack()], epochs=EPOCHS,)



    # 从给定的目录加载一个模型
    def load_model_fromfile(self, model_path):
        crf = tfa.layers.CRF(len(self.labels_category), sparse_target=True)
        return load_model(model_path, custom_objects={"CRF": tfa.layers.CRF, 'crf_loss': crf.loss_function,
                                                      'crf_viterbi_accuracy': crf.accuracy})

    # 预测, 主要用于交互式的测试某些样本的预测结果. 我个人习惯在训练完成之后手动测试一些常见的case,
    def predict(self, model_path, data, maxlen):
        model = self.model
        char2id = [self.vocab.get(i) for i in data]
        input_data = pad_sequences([char2id], maxlen)
        model.load_weights(model_path)
        result = model.predict(input_data)[0][-len(data):]
        result_label = [np.argmax(i) for i in result]
        return result_label

    # 测试
    def test(self, model_path, data, label):
        model = self.load_model_fromfile(model_path)
        loss, acc = model.evaluate(data, label)
        return loss, acc


#训练模型
if __name__ == '__main__':
    if args.mode == 'train':
        train_data = DataSet(args.train_data, CHUNK_TAGS)
        #test_data = DataSet(args.test_data, CHUNK_TAGS)
        w2i = Word2Id(args.char2Id_file)
        if args.gen_vocab:
            w2i.gen_save()
        vocab = w2i.load()
        train_sentence, train_sen_tags = train_data.generate_data(vocab, args.embedding_size)
       #test_sentence, test_sen_tags = test_data.generate_data(vocab,args.embedding_size)
       # print(sentence)
        #print(sen_tags)
        ner = Ner(vocab, CHUNK_TAGS)
        ner.train(train_sentence, train_sen_tags,args.epoch)

    ## 测试模型
    elif args.mode == 'test':
        data = DataSet(args.test_data, CHUNK_TAGS)
        vocab = Word2Id(args.char2Id_file).load()
        sentence, sen_tags = data.generate_data(vocab, args.embedding_size)
        ner = Ner(vocab, CHUNK_TAGS)
        loss, accuracy = ner.test(args.input_model_dir, sentence, sen_tags)
        print("loss: ",loss,"acc: ",accuracy)



    elif args.mode == 'predict':
        vocab = Word2Id(args.char2Id_file).load()
        ner = Ner(vocab, CHUNK_TAGS)
        while (1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                res = ner.predict(args.input_model_dir, demo_sent, args.embedding_size)
                res2label = [CHUNK_TAGS[i] for i in res]
                print(res2label)

    elif args.mode == 'retrain':
        data = DataSet(args.train_data, CHUNK_TAGS)
        vocab = Word2Id(args.char2Id_file).load()
        sentence, sen_tags = data.generate_data(vocab, args.embedding_size)
        ner = Ner(vocab, CHUNK_TAGS)
        ner.retrain(args.input_model_dir, sentence, sen_tags, 2)









