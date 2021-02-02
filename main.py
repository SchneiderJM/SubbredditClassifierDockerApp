from flask import Flask, jsonify, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle

app = Flask(__name__)
@app.route('/')
def getSubreddit():
	outputToSubredditName = {0:'politics',1:'askReddit',2:'worldnews',3:'funny',4:'gaming',5:'aww'}
	stemmer = PorterStemmer()
	stop = set(stopwords.words('english'))

	tfidf = pickle.load(open('./tfidf.p','rb'))
	booster = pickle.load(open('./tfidfbooster.p','rb'))
	testText = pd.DataFrame([request.args.get('query')])[0]
	#Stores this for later to just plug in the input text
	inputText = testText
	#inputText = inputText.map(lambda x: word_tokenize(x.lower()))
	#inputText = inputText.map(lambda x: 
	

	processedText = tfidf.transform(testText)
	output = booster.predict(processedText)
	return(jsonify({'subredditName':outputToSubredditName[output[0]]}))

if __name__ == '__main__':
	app.run(debug=True)
