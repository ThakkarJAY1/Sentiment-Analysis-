from flask import Flask, render_template,redirect,request,url_for,session,jsonify,send_file
from Project1 import test_sample
from Project1 import logreg
from Project1 import model1
from Project1 import model2
from Project1 import RFC
from Project1 import classifier_linear
from Project1 import csv_predict


app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        option = request.form['options']
        option=str(option)
        print("OPTION IS ==== ", option)
        senti=request.form['sentence']
        if option == "logreg":
            flag=test_sample(logreg, senti)
        elif option == "RFC":
            flag=test_sample(RFC, senti)
        elif option == "classifier_linear":
            flag=test_sample(classifier_linear, senti)
        elif option == "file_upload":
            f = request.files['filename']
            csv_predict(f)
            flag=2
        return render_template('index.html',flag=flag,senti=senti)
    else:
        return render_template('index.html',flag=-1)

@app.route('/download')
def download():
	path = "./Predicted.csv"
	return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.secret_key='arun'
    app.run(debug=True)