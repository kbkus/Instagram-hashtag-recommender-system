import flask
import pickle

import pandas as pd
import os

from model.predict import predict
import gensim

import numpy as np
from scipy.spatial.distance import cosine

from werkzeug.utils import secure_filename

# use pickle to load in dataframe of already processed deep features
with open('data/update_processed_data.pkl','rb') as f:
     df = pickle.load(f)
# open dataframe with assigned ratings for each hashtag
with open('data/weighted_tags.pkl','rb') as f:
     ratings = pickle.load(f)

DF = df[['deep_features','tags']]
RATINGS = ratings[['hashtag','rating']]

# create skipgram model
model_skipgram = gensim.models.Word2Vec(
    df['tags'],
    min_count = 1,
    window = 5,
    sg = 1,
    size = 100,
    hs = 1)


def predict_hashtags(input_features, n, df=DF):
    recs = []
    rdf = df.copy()
    rdf['dist'] = rdf['deep_features'].apply(lambda x: cosine(x, input_features))
    matches = rdf.sort_values(by = 'dist')
    matches = matches.head(6)
    
    nearest_tags = set([tag for tags in matches['tags'] for tag in tags])
    # use Word2Vec to find hashtags
    similar_tags = model_skipgram.most_similar(positive = nearest_tags, topn = int(50))
    data = []
    for tag in similar_tags:
        data.append({'hashtag': tag[0],
        'similarity': np.median(RATINGS[RATINGS['hashtag']==tag[0]]['rating'])})
    similar_tags = pd.DataFrame(data)
    top_n = similar_tags.sort_values(by='similarity', ascending=False).reset_index(drop=True).iloc[0:int(n)]

    # create a list of just the tags, not including the similarity values
    for tag in top_n['hashtag']:
        recs.append(tag)
    return recs
     

UPLOAD_FOLDER = 'static/uploads'

#template_folder – the folder that contains the templates that should be used by the application. Defaults to 'templates' folder in the root path of the application.
app = flask.Flask(__name__, template_folder = 'templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# basic function that will route to the hashtag_project.html page when run
@app.route('/',methods=['GET', 'POST'])
def main():
    # If someone simply goes to the webpage, open the html for that webpage
    if flask.request.method == 'GET':
        return(flask.render_template('hashtag_project.html'))
    
    # if someone submits a picture, extract the deep features, run through the model
    # return the input and the predictions
    if flask.request.method == 'POST':
        # get the input picture
        if 'file' not in flask.request.files:
            flask.flash('no file part')
            # if there is no input picture, redirect back to the original url
            return flask.redirect(flask.request.url)
        # get file
        file = flask.request.files['file']
        # get the number of hashtags needed to return
        n = flask.request.form.get('n')
        # create secure file name
        filename = secure_filename(file.filename)
        # save file to uploads folder
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        features = predict(path)
        prediction = predict_hashtags(input_features=features, n=n)
        # get the number of hashtags to recommend from the form
        
        return flask.render_template('hashtag_project.html',
                                    filename = filename,
                                    result = prediction,
                                    )


# Create links that will go to other webpages
# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     return(flask.render_template('home.html'))

# @app.route('/about', methods=['GET', 'POST'])
# def about():
#     return(flask.render_template('about.html'))

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     return(flask.render_template('contact.html'))

# @app.route('/hashtag_project', methods=['GET', 'POST'])
# def hashtag_project():
#     return(flask.render_template('hashtag_project.html'))

# @app.route('/index', methods=['GET', 'POST'])
# def hashtag_project():
#     return(flask.render_template('index.html'))

if __name__ == '__main__':
    app.run()