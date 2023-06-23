from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, All
from werkzeug.utils import secure_filename

app = Flask(__name__)
Bootstrap(app)

# configuration
files = UploadSet('files', All)
app.config['UPLOADED_FILES_DEST']= 'static/uploadstorage'
configure_uploads(app, files)

import os
import datetime
import time

# EDA packages
import pandas as pd
import numpy as np

# ML packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ML Packages for Vectorization of Text for Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/datauploads', methods=['GET', 'POST'])
def datauploads():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploadstorage', filename))

        # Date
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA
        df = pd.read_csv(os.path.join('static/uploadstorage', filename))
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns) 
        df_targetname = df[df.columns[-1]].name
        df_featurenames = df.columns[1:-1]
        # select all columns even the last column
        df_Xfeatures = df.iloc[:, 1:-1]
        # select last column as a target
        df_Ylabels = df[df.columns[-1]]
        # sf_Ylabels = df.iloc[:, -1]

        # Table
        df_table = df

        X = df_Xfeatures
        Y = df_Ylabels

        # MOdel Building
        models = []
        models.append(('LR', LogisticRegression(max_iter=10000)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
        models.append(('CART', DecisionTreeClassifier(max_depth=2)))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(kernel='linear', C=1)))

        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        best_scores = list([])
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(name)
            model_results = results
            model_names = names

    return render_template('details.html', filename=filename, df_table=df_table, 
                           date=date, df_size=df_size, df_shape=df_shape, df_columns=df_columns,
                           df_targetname=df_targetname, model_results=allmodels,
                           model_names=names, df_featurenames=df_featurenames)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
