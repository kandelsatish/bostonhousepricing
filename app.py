import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd



# create a flask app or initialize the flask application
app=Flask(__name__)

# load your model
model=pickle.load(open('regression_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',method=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])



if __name__=="__main__":
    app.run(debug=True)





