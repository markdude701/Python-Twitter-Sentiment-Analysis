import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

###############################
##Importing Features and Docs##
###############################

documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets_f = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
#print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

###################################
##Importing Pickled NB Algorithms##
###################################

open_file = open("originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

################################################
##Turn these on to test the sentiment analysis##
################################################

#user_input = input("Please offer some text to analyze: " )

#print(sentiment(user_input))


##################
##Twitter Set Up##
##################

#consumer key, consumer secret, access token, access secret.
ckey = "X"
csecret = "X"
atoken = "X"
asecret = "X"


class listener(StreamListener):

	def on_data(self, data):
		all_data = json.loads(data)

		tweet = all_data["text"]
		sentiment_value, confidence = sentiment(tweet)
		print(tweet, sentiment_value, confidence)

		if confidence*100 >= 80:
			output = open("twitter-out.txt","a")
			output.write(sentiment_value)
			output.write('\n')
			output.close()

		return True

	def on_error(self, status):
		print(status)
try:
	auth = OAuthHandler(ckey, csecret)
	auth.set_access_token(atoken, asecret)

	twitterStream = Stream(auth, listener())
	twitterStream.filter(track=['happy'])
except Exception:
	pass

###############
##Please Note##
###############
'''
When using Twitter’s streaming API one must be careful of the dangers of rate limiting. If clients exceed a limited number of attempts to connect to the streaming API in a window of time, they will receive error 420. The amount of time a client has to wait after receiving error 420 will increase exponentially each time they make a failed attempt.
'''
