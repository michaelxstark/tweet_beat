from nltk.sentiment import SentimentIntensityAnalyzer

import tweepy
import nltk
import sqlite3

import numpy as np
from scipy import signal as sig
import pygame
import time

nltk.download('vader_lexicon')


# Authentification
consumer_key = 'c_k'  # dummy key
consumer_secret = 'c_s'  # dummy key
access_token = 'a_t'  # dummy key
access_token_secret = 'a_t_s'  # dummy key
bearer_token = 'b_t'  # dummy key


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

api.verify_credentials()

client = tweepy.Client(bearer_token)
client.get_user(username='elonmusk')


# Get tweets from elonmusk
tweets = api.user_timeline(screen_name='elonmusk',
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts=False,
                           # Necessary to keep full_text
                           # otherwise only the first 140 words are extracted
                           tweet_mode='extended'
                           )


# defining audio related stuff


# defining the samplerate

global sr
sr = 44100


# all the instruments

def kick(frq, dur):
    line = np.linspace(0, 1, int((sr / 1000) * dur))
    line2 = np.sqrt(line)
    line3 = line2 * frq - 0.15
    line4 = np.cos(line3)
    envexp = 0.5 ** (25 * line)
    kick = line4 * envexp
    sos = sig.butter(2, 300, 'lp', analog=False, fs=1000, output='sos')
    filtered = sig.sosfilt(sos, kick)
    return filtered


def snare(frq, dur):
    noise = np.random.random_sample(int((sr / 1000) * dur)) * 2 - 1
    line = np.linspace(0, 1, int((sr / 1000) * dur))
    envexp = 0.5 ** (12.5 * line)
    sos = sig.butter(4, 20, 'hp', analog=False, fs=1000, output='sos')
    filtered = sig.sosfilt(sos, (noise * envexp))
    sos = sig.butter(1, [5, 40], 'bp', fs=1000, output='sos')
    filtered_2 = sig.sosfilt(sos, filtered)

    def sine_tone(frq, dur):
        sr = 44100
        line = np.linspace(0, 1, int((sr / 1000) * dur))
        t = np.arange(int((sr / 1000) * dur)) / sr
        envexp = 0.5 ** (25 * line)
        sine = 1 * np.sin(2 * np.pi * frq * t) * envexp
        return sine

    snare = (filtered_2 + sine_tone(frq, dur)) * 4
    return snare


def hi_hat(dur):
    line = np.linspace(1, 0, int((sr / 1000) * dur))
    line2 = line ** 4

    def square_tone(frq, dur):
        sr = 44100
        line = np.linspace(0, 1, int((sr / 1000) * dur))
        t = np.arange(int((sr / 1000) * dur)) / sr
        envexp = 0.5 ** (25 * line)
        sine = 1 * np.sin(2 * np.pi * frq * t)
        square = np.where(sine > 0, 1, -1) * envexp
        return square

    noise = np.random.random_sample(int((sr / 1000) * dur)) * 2 - 1
    high_noise = square_tone(350, dur) + square_tone(800, dur) + (noise / 4)
    sos = sig.butter(10, 100, 'hp', analog=False, fs=1000, output='sos')
    filtered = sig.sosfilt(sos, high_noise)
    sos = sig.butter(2, 100, 'hp', analog=False, fs=1000, output='sos')
    filtered_2 = sig.sosfilt(sos, filtered)
    line3 = filtered_2 * line2 * 4
    return line3


def open_hat(dur):
    line = np.linspace(1, 0, int((sr / 1000) * dur))
    line2 = line ** 0.2

    def square_tone(frq, dur):
        sr = 44100
        line = np.linspace(0, 1, int((sr / 1000) * dur))
        t = np.arange(int((sr / 1000) * dur)) / sr
        envexp = 0.5 ** (25 * line)
        sine = 1 * np.sin(2 * np.pi * frq * t)
        square = np.where(sine > 0, 1, -1) * envexp
        return square

    noise = np.random.random_sample(int((sr / 1000) * dur)) * 2 - 1
    high_noise = square_tone(350, dur) + square_tone(800, dur) + (noise / 4)
    sos = sig.butter(10, 50, 'hp', analog=False, fs=1000, output='sos')
    filtered = sig.sosfilt(sos, high_noise)
    sos = sig.butter(2, 50, 'hp', analog=False, fs=1000, output='sos')
    filtered_2 = sig.sosfilt(sos, filtered)
    line3 = filtered_2 * line2 * 4
    return line3


def wood_block(frq, ratio, amount, dur):
    def sine_tone(frq, dur):
        sr = 44100
        line = np.linspace(0, 1, int((sr / 1000) * dur))
        t = np.arange(int((sr / 1000) * dur)) / sr
        envexp = 0.5 ** (25 * line)
        sine = 1 * np.sin(2 * np.pi * frq * t) * envexp
        return sine
    fm = frq + sine_tone(frq * ratio, dur) * amount
    sr = 44100
    line = np.linspace(0, 1, int((sr / 1000) * dur))
    t = np.arange(int((sr / 1000) * dur)) / sr
    envexp = 0.5 ** (25 * line)
    sine = 1 * np.sin(2 * np.pi * fm * t) * envexp
    return sine


def mid_tom(frq, dur):
    line = np.linspace(1, 0, int((sr / 1000) * dur))
    line2 = line ** 3.5
    freq = np.linspace(np.sqrt(frq + (frq * 0.5)),
                       np.sqrt(frq), int((sr / 1000) * dur))
    freq2 = freq ** 2
    t = np.arange(int((sr / 1000) * dur)) / sr
    sine = 1 * np.sin(2 * np.pi * freq2 * t) * line2 * 2.5
    noise = np.random.random_sample(int((sr / 1000) * dur)) * 2 - 1
    sos = sig.butter(10, 70, 'hp', analog=False, fs=1000, output='sos')
    filtered = sig.sosfilt(sos, noise)
    sos = sig.butter(2, 30, 'lp', analog=False, fs=1000, output='sos')
    filtered_2 = sig.sosfilt(sos, filtered)
    tom = sine + ((filtered_2 * line2) * 0.085)
    return tom


# function for turning arrays in pygame-sound-objects

def conv(soundarr):
    sound = np.array([32767 * soundarr, 32767 * soundarr]).T.astype(np.int16)
    sound = pygame.sndarray.make_sound(sound.copy())
    sound.set_volume(0.1)
    return sound


# initializing the mixer
pygame.mixer.pre_init(44100, size=-16, channels=2)
pygame.mixer.init()
pygame.mixer.set_num_channels(32)


# initialize Sentiment_Analyzer
sia = SentimentIntensityAnalyzer()


# test_score
test_score = sia.polarity_scores(tweets[100].full_text)['compound']
test_score


# creating a test_database for the compound_polarity_scores of musk's tweets
conn = sqlite3.connect('em_test_database')
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS compound_scores
          ([score_id] INTEGER PRIMARY KEY, [score] DOUBLE)
          ''')

conn.commit()

# inserting values
for i, j in enumerate(tweets):
    sco = sia.polarity_scores(j.full_text)['compound']
    c.execute(f'INSERT INTO compound_scores (score_id, score) VALUES ({i}, {sco})')


# the main loop

# defining time between sounds
beat = 0.15

for i in range(len(tweets)):
    val = list(c.execute(f'SELECT score FROM compound_scores WHERE score_id = {i}'))[0][0]
    print(val, end='\r')
    if val == 0.0:
        conv(kick(150, 600) * 0.0).play()
    elif -1.0 <= val <= -0.5:
        conv(kick(150, 600)).play()
    elif -0.5 < val <= 0.1:
        conv(hi_hat(np.random.randint(75, 250))).play()
    elif 0.1 < val <= 0.25:
        conv(snare(200, 300) * 0.7).play()
    elif 0.25 < val <= 0.5:
        conv(wood_block(2000, 10, 2, 200)).play()
    elif 0.5 < val <= 1.0:
        conv(mid_tom(np.random.randint(50, 150), 1000) * 0.2).play()
    time.sleep(beat)
