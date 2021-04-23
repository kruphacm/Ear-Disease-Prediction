import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelear.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLear.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x)-1 for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'Ear Infection Treatment:Ear drops,Ibuprofen and Acetaminophen',1:'Inner ear Infection   Treatment:Vestibular rehabilitation Therapy',2:'Middle Ear Infection   Treatment:Decongestant,nasal steroids or antihistamine',3:'Outer Ear Infection    Treatment:Antibiotics'}
    output=res[output]

    return render_template('HTMLear.html', prediction_text='Predicted Disease is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)