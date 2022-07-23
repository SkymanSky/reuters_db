from cProfile import label
from keras.datasets import reuters
import numpy as np

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
#num_words verilerin en sık karşılaşılan 10.000 kelime ile sınırlı olmasını sağlar.

word_index=reuters.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[2]])
#İndekslerin 3 ten başladığına dikkatiniz çekerim.
#Çünkü 0,1 ve 2 "doldurma","başlangıç" ve "bilinmeyen" için ayrılmıştır.

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

#Eğitim verilerinin vektöre dönüştürülmesi
x_train=vectorize_sequences(train_data)

#Test verilerinin vektöre dönüştürülmesi
x_test=vectorize_sequences(test_data)

#Etiketleri vektöre dönüştürmek için one-hot encoding yapılır. 46 farklı haber kategorisi olduğu için dimension=46
def one_hot_encoding(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label]=1
    return results

#Eğitim ve test etiketlerinin vektöre dönüştürülmesi
one_hot_train_labels=one_hot_encoding(train_labels)
one_hot_test_labels=one_hot_encoding(test_labels)

#Etiketleri vektöre dönüştürmek amacıyla yapılan one-hot encoding işlemini-
#kerasda aşağıdaki hazır fonksiyonla yapabilirsiniz

from keras.utils.np_utils import to_categorical

one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

#Kod 3.15 Model Tanımlama
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

#Kod 3.16 Modeli derlemek
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#Kod 3.17 Doğrulama veri seti oluşturmak
x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

#3.18 Modeli Eğitmek
history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))

#Kod 3.19 Eğitim ve Doğrulama kayıplarını çizdirmek
import matplotlib.pyplot as plt

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Eğitim Kaybı')
plt.plot(epochs,val_loss,'b',label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel='Epoklar'
plt.ylabel='Kayıp'
plt.legend()

plt.show()