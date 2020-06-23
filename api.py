from flask import Flask,request,render_template
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import os
import json
from recognize_face import data_config, test
from finger import test_finger

app = Flask(__name__)
cors = CORS(app)

ds = pd.read_csv('details.csv')
fields = ds.columns
df = ds.values
no_of_fields = len(fields)
fingers = ['thumb','index','middle','ring','little']

data_config()

def generate_demographic_details(index):
  ans = {}
  for j in range(no_of_fields):
    ans[fields[j]]=df[index][j]
  return ans

@app.route('/face',methods=['POST'])
def hello():
  if request.method == 'POST':
    face_image = request.files['face']
    path = 'face_test/{}'.format(face_image.filename)
    face_image.save(path)
    print(path,flush=True)
    index = test(path)
    if index == -1:
      return json.dumps({
        'message': 'There should be only one face in an image'
      }),422
    elif index == -2:
      return json.dumps({
        'message': 'No match found'  
      }),200
    print(index,flush=True)
    ans = generate_demographic_details(index)
    return json.dumps(ans)
  
@app.route('/fingerprint',methods=['POST'])
def get_data():
  if request.method == 'POST':
    fingerprint_image = request.files['fingerprint']
    path = 'fingerprint_test/{}'.format(fingerprint_image.filename)
    fingerprint_image.save(path)
    finger_type = fingers.index(request.form['text'])
    index = test_finger(path,finger_type)
    ans = generate_demographic_details(index)
    return json.dumps(ans)

app.run(port=4000)