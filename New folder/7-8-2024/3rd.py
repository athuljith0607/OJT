#create a model with logistic regression and bow for the text classification

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#create sample data
reviews =[
    "i love the movie,it was good",
    "the movie was boring",
    "excellent movie, actors done well",
    "it was a normal movie, nothing special"
]

#positive=1, negative=0, neutral=2
labels = [1,0,1,2]
#create vectorization for the above
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#create our model
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred =model.predict(x_test)
accuracy =accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")

#create a sample review for the prediction
new_review =["I really enjoyed the movie"]
new_vector =vectorizer.transform(new_review)
prediction =model.predict(new_vector)
predict_class =prediction[0]
sentiment_dict ={1:"positive",0:"negative",2:"neural"}
print(f"predicted sentiment:{sentiment_dict[predict_class]}")
