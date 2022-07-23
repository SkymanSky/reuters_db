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

print(one_hot_test_labels[0])

