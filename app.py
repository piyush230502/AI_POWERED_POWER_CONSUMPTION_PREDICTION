from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 



@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Temperature =float(request.form['Temperature'])
            Humidity =float(request.form['Humidity'])
            Wind_Speed =float(request.form['Wind_Speed'])
            general_diffuse_flows =float(request.form['general_diffuse_flows'])
            diffuse_flows =float(request.form['diffuse_flows'])
            Zone1 =float(request.form['Zone1'])
            Zone2 =float(request.form['Zone2'])
            Zone3 =float(request.form['Zone3'])

            data = [Temperature, Humidity, Wind_Speed, general_diffuse_flows, diffuse_flows, Zone1, Zone2, Zone3]
            data = np.array(data).reshape(1, 17)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')
    



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 5005, debug=True)
