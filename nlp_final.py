#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:00:36 2021

@author: kevinxie
"""

#%%
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt



#%%
dataset = pd.read_json('./public_data/train.json')


# # delete emoji
# def demoji(word):
# 	emoji_pattern = re.compile("["
# 		u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U00010000-\U0010ffff"
# 	                           "]+", flags=re.UNICODE)
# 	return(emoji_pattern.sub(r'', str(word)))

# # delete 超連結
# def dehttp(word):
# 	return re.sub(r"http\S+", "", str(word))
    
# want_lower_column = ['text', 'reply']
# for columns in want_lower_column:
#     dataset[columns] = dataset[columns].str.lower() 
#     dataset[columns] = dataset[columns].str.replace('@', '')
#     dataset[columns] = dataset[columns].str.replace('#', '')
#     dataset[columns] = demoji(dataset[columns].str)
#     dataset[columns] = dataset[columns].apply(lambda x: re.split('https:\/\/,*', str(x))[0])

temp_text = dataset['text'].copy()

for i in range(len(temp_text)):
    temp_text[i] = temp_text[i].lower()
    
    temp_text[i] = temp_text[i].strip()
    
    # remove punctuation
    temp_text[i] = re.sub(r'[^\w\s]', '', temp_text[i])
    
    # remove hyper link
    temp_text[i] = re.sub(r"http\S+", '', temp_text[i])

dataset['text'] = temp_text

train_data, test_data = train_test_split(dataset, test_size=0.2)


#%%

x_train = train_data['text']
y_train = train_data['label']
y_train = y_train.replace('label', 0, regex=True)

x_test = test_data['text']
y_test = test_data['label']
y_true = y_test


#%%
# 建立Token
token = Tokenizer(num_words=10000) #使用Tokenizer模組建立token，建立一個3800字的字典
#讀取所有訓練資料影評，依照每個英文字在訓練資料出現的次數進行排序，前3800名的英文單字會加進字典中
token.fit_on_texts(x_train)
# print(token.word_index) #可以看到它將英文字轉為數字的結果，例如:the轉換成1
#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list
x_train_seq = token.texts_to_sequences(x_train)
x_test_seq = token.texts_to_sequences(x_test) 
# print(x_train_seq)
# print(x_test_seq)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_true = le.fit_transform(y_true)


#%%
# 每一篇影評文字字數不固定，但後續進行深度學習模型訓練時長度必須固定
# 截長補短
# max_train_len = len(max(x_train_seq, key = len))
# max_test_len = len(max(x_test_seq, key = len))

x_train = sequence.pad_sequences(x_train_seq, maxlen = 50)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 50)
#長度小於380的，前面的數字補0 #長度大於380的，截去前面的數字
#變成25000*380的矩陣 = 25000則評論，每則包含380個數字
# print(x_train)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
# print(x_train)

#%%

# RNN

modelRNN = Sequential()
modelRNN.add(Embedding(output_dim=32, #輸出的維度是32，希望將數字list轉換為32維度的向量
                        input_dim=10000, #輸入的維度是3800，也就是我們之前建立的字典是3800字 
                        input_length=50)) #數字list截長補短後都是380個數字


# 建立RNN層，建立16個神經元的RNN層
modelRNN.add(SimpleRNN(units=4))
# 建立隱藏層，建立256個神經元的隱藏層，ReLU激活函數
modelRNN.add(Dense(units=32,activation='relu'))
#隨機在神經網路中放棄70%的神經元，避免overfitting
modelRNN.add(Dropout(0.7)) 
# 建立輸出層，Sigmoid激活函數
modelRNN.add(Dense(units=1,activation='sigmoid')) #建立一個神經元的輸出層
modelRNN.summary()

#%%
# 定義訓練模型
modelRNN.compile(loss='binary_crossentropy',
                 optimizer='adam', 
                 metrics=['accuracy'])

#%%

#Loss function使用Cross entropy 
#adam最優化方法可以更快收斂
train_history = modelRNN.fit(x_train,
                             y_train, 
                             epochs=5,
                             batch_size=1000, 
                             verbose=2, 
                             validation_split=0.2)

#%%

scores = modelRNN.evaluate(x_test, y_true,verbose=1)
print(scores[1])

#%%
# eval pharse


dataset = pd.read_json('./public_data/eval.json')

temp_text = dataset['text'].copy()

for i in range(len(temp_text)):
    temp_text[i] = temp_text[i].lower()
    
    temp_text[i] = temp_text[i].strip()
    
    # remove punctuation
    temp_text[i] = re.sub(r'[^\w\s]', '', temp_text[i])
    
    # remove hyper link
    temp_text[i] = re.sub(r"http\S+", '', temp_text[i])
    

dataset['text'] = temp_text
x_train = dataset['text']
# tokenizer

token = Tokenizer(num_words=10000) #使用Tokenizer模組建立token，建立一個3800字的字典
#讀取所有訓練資料影評，依照每個英文字在訓練資料出現的次數進行排序，前3800名的英文單字會加進字典中
token.fit_on_texts(x_train)
# print(token.word_index) #可以看到它將英文字轉為數字的結果，例如:the轉換成1
#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list
x_train_seq = token.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train_seq, maxlen = 50)
x_train = np.asarray(x_train).astype(np.float32)

# model predict
Y_pred = [1 if o>0.5 else 0 for o in modelRNN.predict(x_train)]
pred_label = pd.DataFrame(Y_pred,columns = ['label'])
#pred_label = pd.Series(le.inverse_transform(pred_label))
pred_label = pred_label.replace({0:'fake',1:'real'})
submit_df = pd.concat([dataset['idx'],dataset['context_idx'],pred_label],axis = 1)



submit = submit_df.to_csv('eval.csv',index = False)





#%%
def show_train_history(train, val, accuracy_or_loss):
    # accuracy_or_loss : input 'Accuracy' or 'loss'
    plt.figure()
    plt.plot(train_history.history[train]) 
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel(accuracy_or_loss)
    plt.legend(["train", "validation"], loc="upper left") 
    plt.show()
    

#%%

show_train_history('accuracy', 'val_accuracy', 'Accuracy')
show_train_history('loss', 'val_loss', 'Loss')


#%%

#LSTM

modelLSTM = Sequential() #建立模型
modelLSTM .add(Embedding(output_dim=32, #輸出的維度是32，希望將數字list轉換為32維度的向量
                         input_dim=10000, #輸入的維度是3800，也就是我們之前建立的字典是3800字
                         input_length=50)) #數字list截長補短後都是380個數字

# 建立LSTM層 
modelLSTM .add(LSTM(4)) #建立32個神經元的LSTM層
# 建立隱藏層
modelLSTM .add(Dense(units=32,activation='relu')) #建立256個神經元的隱藏層
modelLSTM .add(Dropout(0.7))
# 建立輸出層，建立一個神經元的輸出層
modelLSTM .add(Dense(units=1,activation='sigmoid'))
# 查看模型摘要
modelLSTM .summary()


#%%

modelLSTM.compile(loss='binary_crossentropy',
                 optimizer='adam', 
                 metrics=['accuracy'])
#Loss function使用Cross entropy 
#adam最優化方法可以更快收斂
train_history = modelLSTM.fit(x_train,
                             y_train, 
                             epochs=5,
                             batch_size=1000, 
                             verbose=2, 
                             validation_split=0.2)

#%%

scores = modelLSTM.evaluate(x_test, y_true, verbose=1)
print(scores[1])

show_train_history('accuracy', 'val_accuracy', 'Accuracy')
show_train_history('loss', 'val_loss', 'Loss')

#%%
# eval pharse


dataset = pd.read_json('./public_data/eval.json')

temp_text = dataset['text'].copy()

for i in range(len(temp_text)):
    temp_text[i] = temp_text[i].lower()
    
    temp_text[i] = temp_text[i].strip()
    
    # remove punctuation
    temp_text[i] = re.sub(r'[^\w\s]', '', temp_text[i])
    
    # remove hyper link
    temp_text[i] = re.sub(r"http\S+", '', temp_text[i])
    

dataset['text'] = temp_text
x_train = dataset['text']
# tokenizer

token = Tokenizer(num_words=10000) #使用Tokenizer模組建立token，建立一個3800字的字典
#讀取所有訓練資料影評，依照每個英文字在訓練資料出現的次數進行排序，前3800名的英文單字會加進字典中
token.fit_on_texts(x_train)
# print(token.word_index) #可以看到它將英文字轉為數字的結果，例如:the轉換成1
#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list
x_train_seq = token.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train_seq, maxlen = 50)
x_train = np.asarray(x_train).astype(np.float32)

# model predict
Y_pred = [1 if o>0.5 else 0 for o in modelRNN.predict(x_train)]
pred_label = pd.DataFrame(Y_pred,columns = ['label'])
#pred_label = pd.Series(le.inverse_transform(pred_label))
pred_label = pred_label.replace({0:'fake',1:'real'})
submit_df = pd.concat([dataset['idx'],dataset['context_idx'],pred_label],axis = 1)



submit = submit_df.to_csv('eval.csv',index = False)
