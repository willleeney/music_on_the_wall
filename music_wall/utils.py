import io
import os
import cv2
import time
import sys
import argparse

import requests
import tekore as tk
import pickle
import json
import webbrowser
import tekore as tk
from urllib.parse import urlparse, parse_qs
import sys

import numpy as np

import matplotlib.pyplot as plt


from google.cloud import vision


class Parser:
    """
    Command line parser for the training
    """
    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--capture', type=str, default='video_cam',
                            help='What type of detection of imgs to use (Options: img_file, video_cam)')
        parser.add_argument('--model', type=str, default='words',
                            help='What format to detect key words in (Options: words, album, gesture, holistic)')
        parser.add_argument('--account', type=str, default='2',
                            help='account select override if you know your account (Options: None, 1,2,3)')

        self.parser = parser

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args

def load_configs(args):
    return args

def set_creds():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './music project-d58b0409e659.json'
    return

def spotify_login(choice=None):

    #if len(sys.argv) > 1:
       # chosen_account = str(sys.argv[1])

    #else:
    account_cfg_files = os.listdir('accounts_cfg')
    if choice is None:
        print('Please choose an account or add a new one.')
        print('\n')
        for idx, file in enumerate(account_cfg_files):
            if file[-4:] == '.cfg':
                file_name = file[:-4]
                print(f'{idx}) - {file_name}', )
        print('\n')
        choice = input("Type an account number or 'new' for a new account: ")

        try:
            chosen_account = account_cfg_files[int(choice)]

        except ValueError:
            if choice == 'new':
                account_name = input('Please enter a name for your new account: ')
                client_id = input('Please input your client ID: ')
                client_secret = input('Please input your client secret: ')
                redirect_uri = 'https://example.com/callback'  # Or your redirect uri
                conf = (client_id, client_secret, redirect_uri)
                chosen_account = f'{account_name}.cfg'

                token = tk.prompt_for_user_token(*conf, scope=tk.scope.every)
                tk.config_to_file(chosen_account, conf + (token.refresh_token,))

            else:
                print("you're a bitch, taking first account")
                chosen_account = account_cfg_files[0]
    else:
        chosen_account = account_cfg_files[int(choice)-1]

    chosen_account = './accounts_cfg/' + chosen_account
    print(f'Loading account from {chosen_account}')

    conf = tk.config_from_file(chosen_account, return_refresh=True)
    print('Refreshing token')
    user_token = tk.refresh_user_token(*conf[:2], conf[3])

    spotify = tk.Spotify(user_token)
    user = spotify.current_user()
    print(f"Connected to {user.display_name}'s account")

    return spotify

def open_capture():

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    # Instantiates a client
    #client = vision.ImageAnnotatorClient()

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            # if q button is pressed then save image
            filepath = os.path.abspath('../wall-music/resources/test.png')
            cv2.imwrite(filepath, frame)

            time.sleep(5)
            break

    vc.release()
    cv2.destroyWindow("preview")
    return filepath


def detect_web(path):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient()
    best_guess = []
    words = []

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))
            best_guess.append(label.label)

    if annotations.web_entities:
        print('\n{} Web entities found: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('\n\tScore      : {}'.format(entity.score))
            print(u'\tDescription: {}'.format(entity.description))
            words.append(entity.description)


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return best_guess, words

def find_best_album(best_label_guess,results,spotify):
    found_match = False
    image_results_lower = [i.lower() for i in results]
    image_results_lower  = best_label_guess + image_results_lower
    #search for all predicted phrases for a matching album and plays that
    for img_res_idx, img_result in enumerate(image_results_lower):
        print(f'Searching for word {img_res_idx+1} of prediction - {img_result}')
        
        #searches spotify for albums with the first predicted word
        search_results = spotify.search(img_result,limit=10,types=('album','artist'))
        search_album_names = [album.name.lower() for album in search_results[0].items]
        
        #loops through predicted words to check if any returned albums have the same name
        for img_result in image_results_lower:
            #print(img_result)
            # check if any of predicted results are in the returned albums list 
            # higher confidence answers are checked first
            try:
                album_idx = search_album_names.index(img_result)
                album_to_play_object = search_results[0].items[album_idx]
                print(f'Matched an album: {album_to_play_object.name}')
                found_match = True
                break

            except:
                pass
                
        if found_match:
            break

    if not found_match:
        album_to_play_object = None
        print('No matching album found')

    return album_to_play_object

