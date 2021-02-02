FROM python:3.7

ADD main.py .
ADD tfidf.p .
ADD tfidfbooster.p .

#Installing dependancies
RUN pip install nltk sklearn pandas xgboost flask

#Downloading nltk components
RUN python -m nltk.downloader stopwords

cmd ["python","main.py"]
