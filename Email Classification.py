import pandas as pd
from nltk.corpus import stopwords
from nltk import stem
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer

data = pd.read_csv("spam.csv")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})

ps = PorterStemmer()
stopwords = set(stopwords.words('english'))

def review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    msg = msg.split()
    # removing stopwords
    msg = [word for word in msg if word not in stopwords]
    # using a stemmer
    msg = " ".join([ps.stem(word) for word in msg])
    return msg

# Processing text messages
data['text'] = data['text'].apply(review_messages)

# train test split 
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)

# training vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# training the classifier 
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)

# testing against testing set 
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test) 
cm= confusion_matrix(y_test, y_pred)

