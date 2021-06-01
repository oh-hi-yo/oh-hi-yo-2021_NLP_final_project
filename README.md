# oh-hi-yo-2021_NLP_final_project

## Topic : The Fake-EmoReact 2021 Competition 
With the development of the Internet,  misinformation gains a lot of attention in research fields and social issues.  Benefit from the convenience of sharing information on online platforms, fake news could be spread rapidly from one to another, especially on social media.  The misinformation could seriously influence modern life in many ways. The situation is even worse, due to the pandemic of Covid-19. Seeing the high impact of fake news on society, we eager to mitigate the effect of fake news by applying NLP techniques.


        [competition](http://blog.csdn.net/guodongxiaren)  
        [leaderboards](https://competitions.codalab.org/competitions/31180?secret_key=2f97f399-8bba-4ed5-a0b7-99e17df1fe1b#learn_the_details)


## Data

The dataset is split into the following three files:
* train.json: 168,521 samples with gold labels, to be used for training the model.
* dev.json: 40,487 unlabeled samples used for practice.
* eval.json:  evaluation dataset will be released at the beginning of the competition. 


## Solution

1. text preprocessing
    >> take away emoji and https
    >> remove punctuation
    >> split validation set from training set
    >> tokenize and using texts_to_sequences to transfer text to sequence
    >> padding sequence

2. training model 
    >> RNN, LSTM

3. predict the label for eval.json


## Usage

```

python3 nlp_final.py

```
