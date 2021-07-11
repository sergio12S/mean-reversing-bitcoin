from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import json
from textblob import TextBlob


access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""


class StdOutListener(StreamListener):
    def on_data(self, data):
        m = json.loads(data.encode('utf-8'))
        if 'lang' not in m:
            return
        if m['lang'] != 'en':
            return

        sentiment = TextBlob(m['text']).sentiment_assessments
        if sentiment.polarity != 0:
            send_data = {
                'created_at': m['created_at'],
                'text': m['text'],
                'source': m['source'],
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'assessments': sentiment.assessments
            }

            message = json.dumps(send_data, indent=2).encode('utf-8')
            print(message)

            return True

    def on_error(self, status_code):
        if status_code == 420:
            return False


if __name__ == '__main__':
    try:
        listen = StdOutListener()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        stream = Stream(auth, listen)
        stream.filter(track=["bitcoin"])
    except Exception as e:
        print(e)
