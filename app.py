from flask import Flask,render_template,url_for,request
from sklearn import preprocessing
import pandas as pd 
import pickle
import joblib
import numpy as np
# In this datasets no need of preprocessing as it's a classification and all are one Hot Encoder type of dataset.
training = pd.read_csv("Training.csv")
cols = training.columns
cols=cols[:-1]
X = training[cols]
y = training["prognosis"]
# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

ytb_model = open("model_F.pkl","rb")
dclf = pickle.load(ytb_model)
#clf = joblib.load("model.joblib")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  if (request.method == 'POST') :
    comment = request.form['comment']
    vect = comment.split(",")
    if len(vect) >= 2 :
      t_x = np.zeros(132)
      for i in range(len(cols)):
        for j in range(len(vect)):
          if vect[j] == cols[i]:
            t_x[i] = 1
      my_prediction = dclf.predict([t_x])
      my_pred = le.inverse_transform(my_prediction)
    else:
      my_pred = "your chances of having any disease are less."
  return render_template('result.html',prediction = my_pred)


if __name__ == '__main__':
  app.run(debug=True, use_reloader=False)