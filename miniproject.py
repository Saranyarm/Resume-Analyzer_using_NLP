import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tika import parser
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

def clean_resume(resumeText):
    resumeText =  resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc ', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def get_filename():
  return r'target.pdf'

def engine(filename, clf, word_vectorizer, le):
  file_data = parser.from_file(filename)
  text_data = clean_resume(file_data['content'])
  text_data = word_vectorizer.transform(np.array([text_data]))
  prediction = clf.predict(text_data)
  return le.inverse_transform(prediction)

def model_run():
  clf=pickle.load(open('model_classifier.pkl', 'rb'))
  word_vectorizer=pickle.load(open('model_vectorizer.pkl', 'rb'))
  le=pickle.load(open('model_label.pkl', 'rb'))
  filename = get_filename()
  return engine(filename, clf, word_vectorizer, le)


UPLOAD_FOLDER = r'C:\Users\Vinesh\Documents\miniproject'
ALLOWED_EXTENSIONS = {'pdf','docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
  return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
  file = request.files['file']
  # If the user does not select a file, the browser submits an
  # empty file without a filename.
  # if file.filename == '':
  #   flash('No selected file')
  #   return redirect(request.url)
  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'target.pdf'))
      # return redirect(url_for('download_file', name=filename))
  output = model_run()
  # int_features = [float(x) for x in request.form.values()]
  # final_features = [np.array(int_features)]
  # prediction = model.predict(final_features)
  # output = round(prediction[0],2)
  # output = "hi"
  return render_template('index.html', output='The Uploaded resume is suitable for {}'.format(output))
if __name__ =="__main__":
  app.run(debug=True)


