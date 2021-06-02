import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup
app = Flask(__name__)
model = pickle.load(open('modelear.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLear.html')

@app.route('/predict',methods=['POST'])
def predict():
    res1={'Mild pain or Discomfort inside the ear':1.0,'A feeling of Pressure inside the ear':2.0,'Pus':3.0,'Hearing loss':4.0,'Dizziness':5.0,"Nausea":6.0,'Vomiting':7.0,'Eardrum Bulge':8.0,'Ear ache':9.0,'Fluid drainage from the ear':10.0,'Painful':11.0,'Tender':12.0,'Red':13.0,'Swollen':14.0}
    int_features = [res1[x] for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'<p>Predicted Disease is Ear Infection<br><br>Treatment:Ear drops,Ibuprofen and Acetaminophen</p>',1:'<p>Predicted Disease is Inner ear Infection<br><br>Treatment:Vestibular rehabilitation Therapy</p>',2:'<p>Predicted Disease is Middle Ear Infection<br><br>Treatment:Decongestant,nasal steroids or antihistamine</p>',3:'<p>Predicted Disease is Outer Ear Infection<br><br>Treatment:Antibiotics</p>'}
    output=res[output]
    output=Markup(output)
    return render_template('HTMLear.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
