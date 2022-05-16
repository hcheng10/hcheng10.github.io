---
layout: post
title: Fake News Classification by NLP
---
Blog Post: Fake News Classification

Rampant misinformation—often called “fake news”—is one of the defining features of contemporary democratic life. In this Blog Post, we will develop and assess a fake news classifier using Tensorflow.

## Data Source
Our data for this assignment comes from the article

Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).

[data: Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 1 Acquire Training Data

The data been hosted at the below URL:


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

# requires update to tensorflow 2.4
# >>> conda activate PIC16B
# >>> pip install tensorflow==2.4
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for embedding viz
import plotly.express as px 
import plotly.io as pio
pio.templates.default = "plotly_white"
```


```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
```


```python
df_train = pd.read_csv(train_url)
df_train.head()
```





  <div id="df-82eaba79-25d2-429b-88e3-304507370626">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-82eaba79-25d2-429b-88e3-304507370626')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-82eaba79-25d2-429b-88e3-304507370626 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-82eaba79-25d2-429b-88e3-304507370626');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
le = LabelEncoder()
df_train["fake"] = le.fit_transform(df_train["fake"])
num_fake = len(df_train["fake"].unique())
```

- Each row of the data corresponds to an article. 
- The title column gives the title of the article, while the text column gives the full article text. 
- The final column, called fake, is 0 if the article is true and 1 if the article contains fake news, as determined by the authors of the paper above.

## 2. Make a Dataset

Write a function called make_dataset. This function should do two things:

- Remove stopwords from the article text and title. A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but.”  Helpful link: [StackOverFlow thread](https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe)
- Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), and the output should consist only of the fake column. [This tutorial](https://www.tensorflow.org/guide/keras/functional) for reference on how to construct and use Datasets with multiple inputs.


```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english') 

'''
function takes a data frame as input, then remove stop words of the dataframe with column names title and text.
next the function use processed dataframe to make a rf.data.dataset object and return it.
'''
def make_dataset(df):
  # remove stopwords for column 
  # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
  df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  data = tf.data.Dataset.from_tensor_slices(( 
      # dictionary for input data/features
      {'title': df[['title']],
       'text': df[['text']]},
       # dictionary for output data/labels
      {'fake': df['fake']}
      ))
  
  return data
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
df_new = make_dataset(df_train)
```

### validation data

split of 20% of the new dataset we made to use for validation.


```python
df_new = df_new.shuffle(buffer_size = len(df_new))

train_size = int(0.8*len(df_new)) # 80% training size
val_size   = int(0.2*len(df_new)) # 20% validation size

train = df_new.take(train_size).batch(20) # grouping into 20s, makes trainning faster
val = df_new.skip(train_size).take(val_size).batch(20)

print(len(train), len(val))
```

    898 225
    

### Base Rate

Recall that the base rate refers to the accuracy of a model that always makes the same guess (for example, such a model might always say “fake news!”). The base rate for this data set by examining the labels on the training set.

When we determine wether a news is fake news or not, without any new or interesting occurs to impact the outcome.​ The rate of a news is a fake news is 52% on the trainning data set which the base rate for this model is **52%**.


```python
sum(df_train["fake"] == 1)/len(df_train)
```




    0.522963160942581



### TextVectorization

Here is one option:


```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int', # get frequency ranking for each word in the training dataset
    output_sequence_length=500) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
```


```python
title_input = keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)

text_input = keras.Input(
    shape=(1, ),
    name = "text",
    dtype = "string"
)
```

## 3. Create Models

Use TensorFlow models to offer a perspective on the following question:

> When detecting fake news, is it most effective to focus on only the title of the article, the full text of the article, or both?

To address this question, create three (3) TensorFlow models.

1. In the first model, use only the article title as an input.
2. In the second model, use only the article text as an input.
3. In the third model, use both the article title and the article text as input.

Train our models on the training data until they appear to be “fully” trained. Assess and compare their performance. Make sure to include a visualization of the training histories.

We can visualize our models with this code:

> ***from tensorflow.keras import utils*** <br>
> ***utils.plot_model(model)***

### Notes

- For the first two models, we don’t have to create new Datasets. Instead, just specify the inputs to the keras.Model appropriately, and TensorFlow will automatically ignore the unused inputs in the Dataset.
- The lecture notes and tutorials linked above are likely to be helpful as we are creating our models as well.
- **We will need to use the Functional API, rather than the Sequential API, for this modeling task.**
- When using the Functional API, it is possible to use the same layer in multiple parts of our model; see this tutorial for examples. I recommended that we share an embedding layer for both the article title and text inputs.
We may encounter overfitting, in which case Dropout layers can help.

We’re free to be creative when designing our models. If we’re feeling very stuck, start with some of the pipelines for processing text that we’ve seen in lecture, and iterate from there. Please include in our discussion some of the things that we tried and how we determined the models we used.

**What Accuracy Should We Aim For?**

Our three different models might have noticeably different performance. **Our best model should be able to consistently score at least 97% validation accuracy.**

After comparing the performance of each model on validation data, make a recommendation regarding the question at the beginning of this section. Should algorithms use the title, the text, or both when seeking to detect fake news?


```python
# title layer
title_features = title_vectorize_layer(title_input) # apply this "function TextVectorization layer" to lyrics_input
title_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding1")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)

# for model1 (title input only)
title_features= layers.Dense(32, activation='relu')(title_features)
output1 = layers.Dense(num_fake , name="fake")(title_features) 

# text layer
text_features = title_vectorize_layer(text_input) # apply this "function TextVectorization layer" to lyrics_input
text_features = layers.Embedding(size_vocabulary, output_dim = 2, name="embedding2")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)

# for model2 (text input only)
text_features= layers.Dense(32, activation='relu')(text_features)
output2 = layers.Dense(num_fake , name="fake")(text_features) 

# for model3 (both title and text)
main = layers.concatenate([title_features, text_features], axis = 1)
main = layers.Dense(32, activation='relu')(main)
output3 = layers.Dense(num_fake, name="fake")(main) 
```


```python
model1 = keras.Model(
    inputs = title_input,
    outputs = output1
)

model2 = keras.Model(
    inputs = text_input,
    outputs = output2
)

model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output3
)
```

### model 1


```python
model1.summary()
```

    Model: "model_9"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization_2 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding1 (Embedding)      (None, 500, 2)            4000      
                                                                     
     dropout_12 (Dropout)        (None, 500, 2)            0         
                                                                     
     global_average_pooling1d_6   (None, 2)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_13 (Dropout)        (None, 2)                 0         
                                                                     
     dense_15 (Dense)            (None, 32)                96        
                                                                     
     dense_16 (Dense)            (None, 32)                1056      
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 5,218
    Trainable params: 5,218
    Non-trainable params: 0
    _________________________________________________________________
    


```python
keras.utils.plot_model(model1)
```




    
![png]({{ site.baseurl }}/images/hw4_pic/output_22_0.png)
    




```python
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history1 = model1.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```

    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
    
    


```python
table1 = pd.DataFrame({'accuracy' : history1.history["accuracy"], 'val_accuracy' : history1.history["val_accuracy"]})    
table1[29:] # last 20 epochs
```





  <div id="df-d6b9fab6-bb1b-45ec-812d-e59878b6476f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.965031</td>
      <td>0.995099</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.960354</td>
      <td>0.993317</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.961523</td>
      <td>0.989753</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.964308</td>
      <td>0.988416</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.962860</td>
      <td>0.984184</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.961078</td>
      <td>0.991980</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.964196</td>
      <td>0.994208</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.964753</td>
      <td>0.992426</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.964308</td>
      <td>0.993317</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.963862</td>
      <td>0.992649</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.967537</td>
      <td>0.986411</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.963974</td>
      <td>0.993985</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.964141</td>
      <td>0.987302</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.963584</td>
      <td>0.994876</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.965811</td>
      <td>0.993763</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.965087</td>
      <td>0.984629</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.963639</td>
      <td>0.988416</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.965310</td>
      <td>0.988862</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.963194</td>
      <td>0.990198</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.965978</td>
      <td>0.993540</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.965644</td>
      <td>0.995099</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d6b9fab6-bb1b-45ec-812d-e59878b6476f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d6b9fab6-bb1b-45ec-812d-e59878b6476f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d6b9fab6-bb1b-45ec-812d-e59878b6476f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
from matplotlib import pyplot as plt
plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7fb300037b10>]




    
![png]({{ site.baseurl }}/images/hw4_pic/output_26_1.png)
    


### model 2


```python
model2.summary()
```

    Model: "model_10"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization_2 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding2 (Embedding)      (None, 500, 2)            4000      
                                                                     
     dropout_14 (Dropout)        (None, 500, 2)            0         
                                                                     
     global_average_pooling1d_7   (None, 2)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_15 (Dropout)        (None, 2)                 0         
                                                                     
     dense_17 (Dense)            (None, 32)                96        
                                                                     
     dense_18 (Dense)            (None, 32)                1056      
                                                                     
     fake (Dense)                (None, 2)                 66        
                                                                     
    =================================================================
    Total params: 5,218
    Trainable params: 5,218
    Non-trainable params: 0
    _________________________________________________________________
    


```python
keras.utils.plot_model(model2)
```




    
![png]({{ site.baseurl }}/images/hw4_pic/output_29_0.png)
    




```python
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history2 = model2.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```

    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
    
    


```python
table2 = pd.DataFrame({'accuracy' : history2.history["accuracy"], 'val_accuracy' : history2.history["val_accuracy"]})    
table2[29:]  # last 20 epochs
```





  <div id="df-0bccff77-0dcb-4a09-8df5-8642a1fc6436">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.965755</td>
      <td>0.991980</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.965477</td>
      <td>0.991980</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.965644</td>
      <td>0.994654</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.964809</td>
      <td>0.994208</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.966312</td>
      <td>0.996213</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.964029</td>
      <td>0.995322</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.966813</td>
      <td>0.994876</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.965254</td>
      <td>0.993763</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.966368</td>
      <td>0.993540</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.966312</td>
      <td>0.994876</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.963806</td>
      <td>0.992649</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.968094</td>
      <td>0.972822</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.965811</td>
      <td>0.993094</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.964976</td>
      <td>0.996213</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.966869</td>
      <td>0.991758</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.965811</td>
      <td>0.993094</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.965922</td>
      <td>0.995767</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.965310</td>
      <td>0.992649</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.966702</td>
      <td>0.995990</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.968150</td>
      <td>0.995322</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.964642</td>
      <td>0.996659</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0bccff77-0dcb-4a09-8df5-8642a1fc6436')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0bccff77-0dcb-4a09-8df5-8642a1fc6436 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0bccff77-0dcb-4a09-8df5-8642a1fc6436');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.plot(history2.history["accuracy"])
plt.plot(history2.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7fb2fcfe3210>]




    
![png]({{ site.baseurl }}/images/hw4_pic/output_33_1.png)
    


### model 3


```python
model3.summary()
```

    Model: "model_11"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     title (InputLayer)             [(None, 1)]          0           []                               
                                                                                                      
     text (InputLayer)              [(None, 1)]          0           []                               
                                                                                                      
     text_vectorization_2 (TextVect  (None, 500)         0           ['title[0][0]',                  
     orization)                                                       'text[0][0]']                   
                                                                                                      
     embedding1 (Embedding)         (None, 500, 2)       4000        ['text_vectorization_2[0][0]']   
                                                                                                      
     embedding2 (Embedding)         (None, 500, 2)       4000        ['text_vectorization_2[1][0]']   
                                                                                                      
     dropout_12 (Dropout)           (None, 500, 2)       0           ['embedding1[0][0]']             
                                                                                                      
     dropout_14 (Dropout)           (None, 500, 2)       0           ['embedding2[0][0]']             
                                                                                                      
     global_average_pooling1d_6 (Gl  (None, 2)           0           ['dropout_12[0][0]']             
     obalAveragePooling1D)                                                                            
                                                                                                      
     global_average_pooling1d_7 (Gl  (None, 2)           0           ['dropout_14[0][0]']             
     obalAveragePooling1D)                                                                            
                                                                                                      
     dropout_13 (Dropout)           (None, 2)            0           ['global_average_pooling1d_6[0][0
                                                                     ]']                              
                                                                                                      
     dropout_15 (Dropout)           (None, 2)            0           ['global_average_pooling1d_7[0][0
                                                                     ]']                              
                                                                                                      
     dense_15 (Dense)               (None, 32)           96          ['dropout_13[0][0]']             
                                                                                                      
     dense_17 (Dense)               (None, 32)           96          ['dropout_15[0][0]']             
                                                                                                      
     dense_16 (Dense)               (None, 32)           1056        ['dense_15[0][0]']               
                                                                                                      
     dense_18 (Dense)               (None, 32)           1056        ['dense_17[0][0]']               
                                                                                                      
     concatenate_3 (Concatenate)    (None, 64)           0           ['dense_16[0][0]',               
                                                                      'dense_18[0][0]']               
                                                                                                      
     dense_19 (Dense)               (None, 32)           2080        ['concatenate_3[0][0]']          
                                                                                                      
     fake (Dense)                   (None, 2)            66          ['dense_19[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 12,450
    Trainable params: 12,450
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
keras.utils.plot_model(model3)
```




    
![png]({{ site.baseurl }}/images/hw4_pic/output_36_0.png)
    




```python
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
history3 = model3.fit(train, 
                    validation_data=val,
                    epochs = 50, 
                    verbose = False)
```


```python
table3 = pd.DataFrame({'accuracy' : history3.history["accuracy"], 'val_accuracy' : history3.history["val_accuracy"]})    
table3[29:] # last 20 epochs
```





  <div id="df-4e5dc206-e6a8-4f6e-a28e-941479f69ed5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.995601</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.996381</td>
      <td>0.999554</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.995490</td>
      <td>0.999777</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.995880</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.995434</td>
      <td>0.999332</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.995434</td>
      <td>0.999777</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.995768</td>
      <td>0.999777</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.996214</td>
      <td>0.999332</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.996158</td>
      <td>0.999554</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.996492</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.996882</td>
      <td>0.998886</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.995824</td>
      <td>0.993094</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.995991</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.996214</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.997550</td>
      <td>0.999777</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.996047</td>
      <td>0.999554</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.996882</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.995768</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.996269</td>
      <td>0.999777</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.997049</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.996436</td>
      <td>0.999554</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4e5dc206-e6a8-4f6e-a28e-941479f69ed5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4e5dc206-e6a8-4f6e-a28e-941479f69ed5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4e5dc206-e6a8-4f6e-a28e-941479f69ed5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
```




    [<matplotlib.lines.Line2D at 0x7fb2fc470490>]




    
![png]({{ site.baseurl }}/images/hw4_pic/output_40_1.png)
    


Conclusion: 

All three models performed well on validation data, they all resulted in more than 99% accuracy on validation data. Model3 used both the title and the text as input, the result is really close to 100%. For simplicity, I would recommend model1 or model2, but if we only focus on the performance result, the model3 is the best based on the output.

## 4. Model Evaluation

Test the model performance on unseen test data.


```python
df_test = pd.read_csv("https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true")
df_test.head()
```





  <div id="df-55956b58-7cb5-4847-9d06-a751eb0c5da1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>420</td>
      <td>CNN And MSNBC Destroy Trump, Black Out His Fa...</td>
      <td>Donald Trump practically does something to cri...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14902</td>
      <td>Exclusive: Kremlin tells companies to deliver ...</td>
      <td>The Kremlin wants good news.  The Russian lead...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>322</td>
      <td>Golden State Warriors Coach Just WRECKED Trum...</td>
      <td>On Saturday, the man we re forced to call  Pre...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16108</td>
      <td>Putin opens monument to Stalin's victims, diss...</td>
      <td>President Vladimir Putin inaugurated a monumen...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10304</td>
      <td>BREAKING: DNC HACKER FIRED For Bank Fraud…Blam...</td>
      <td>Apparently breaking the law and scamming the g...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-55956b58-7cb5-4847-9d06-a751eb0c5da1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-55956b58-7cb5-4847-9d06-a751eb0c5da1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-55956b58-7cb5-4847-9d06-a751eb0c5da1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test = make_dataset(df_test)
test = test.batch(20)
model3.evaluate(test)
```

    1123/1123 [==============================] - 4s 4ms/step - loss: 0.0257 - accuracy: 0.9946
    




    [0.025698255747556686, 0.994565486907959]



The average accuracy for testing data is 99%, which is really good.

## 5. Embedding Visualization


```python
weights = model3.get_layer('embedding2').get_weights()[0] # get the weights from the embedding layer
vocab = title_vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```


```python
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                # size_max = 2,
                 hover_name = "word")

fig.show()
```

{% include plotly_hw4.html %}

From the visualization, we can see a big cluster which means all those words could be found either in true news or fake news. The words near the center of the graph are the most common words in the news such as **'services', 'everyone', and 'bad'** which may not be helpful for the model to determine whether the news is true or not. Also, in the graph, we can also see some outliers on the left or the right of the cluster such as **'gop', 'gov', 'its'**. Those outliers may indicate the news with these words is a trend to be fake or true news.
