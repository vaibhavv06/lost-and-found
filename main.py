from flask import Flask,request,render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__,
        static_url_path='',
        static_folder='static',
        template_folder='templates'
      )
cors = CORS(app)

@app.route('/',methods=['GET'])
def index():
  return render_template('index.html')

app.run(port=7000)