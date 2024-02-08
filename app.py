from flask import Flask, render_template,request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.custom_prediction import PredictPipeline



app = Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train',methods=['GET'])
def trainpage():
    os.system("python custom_main.py")
    return "Successfully trained"
@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            crypto = request.form['crypto']
            date = request.form['date']
            
            predict = PredictPipeline()
            predict = predict.predict(crypto, date)

            return render_template('results.html', prediction = str(predict), crypto=crypto, date=date)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__=="__main__":
    # app.run(host="0.0.0.0",port= 8080, debug=True)
    app.run(host="0.0.0.0",port= 8080)
