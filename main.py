import io
import os
import cv2
import time
import torch


# Imports the Google Cloud client library
from google.cloud import vision
# Explicitly use service account credentials by specifying the private key
# file.
import music_wall.utils as utils
import music_wall.detection as detection
from  music_wall.gesture import SimpleGestureDetector

import tekore as tk



import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def run():
    # parse arguments given
    parser = utils.Parser()
    args = parser.parse()
    # convert arguments into config dictionary
    config = utils.load_configs(args)

    # set google cloud credientials
    utils.set_creds()
    print('acc',config.account)
    # login to spotify and create object for api queries
    spotify = utils.spotify_login(choice=config.account)

    # options to search with image file or from video camera
    if config.capture == 'img_file':
        if config.model == 'writing':
            path = os.path.abspath('../wall-music/resources/test_photo_2-1.jpeg')
        elif config.model == 'album':
            path = os.path.abspath('../wall-music/resources/ultimate_test.jpg')



    # detect writing or key words from album cover
    if config.model == 'words':
        path = utils.open_capture()
        key_words = detection.detect_writing(path)
        # find search query based on key words
        search_query = ' '.join(key_words)
        # search spotify for response to the search query
        #response = spotify.search(search_query)
        #search_response = response[0].items[0]
        #print(f'Playing {search_response.name}')
        #spotify.playback_start_tracks([search_response.id])


        artists, = spotify.search(search_query, types=('artist',), limit=1)
        artist = artists.items[0]
        albums = spotify.artist_albums(artist.id)
        album_uri = albums.items[0].uri
        spotify.playback_start_context(album_uri)




    elif config.model == 'album':
        path = utils.open_capture()
        best_label_guess, words = detection.detect_web(path)
        album_object = utils.find_best_album(best_label_guess,words,spotify)

        if album_object:
            spotify.playback_start_context(album_object.uri)

        #total_squares = utils.detect_squares(path)

        #img  = cv2.imread(path)
        #imgs_found = utils.bound_detect(img)
        #print(f'Images Found: {imgs_found}')

        #squares = utils.find_squares(img)
        #cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
        #filepath = os.path.abspath('../wall-music/resources/test_squares.png')
        #cv2.imwrite(filepath, img)

        detection.localize_objects(path)

        best_label_guess, words = detection.detect_web(path)
        print('stop')


    elif config.model=='gesture':
        # drawing utils
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        # gesture recognition
        hand_dectector = SimpleGestureDetector()

        # For webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            #image.flags.writeable = False
            # get hand dector images
            detection_info = hand_dectector.detectHands(image)
            # draw landmarks
            if detection_info.multi_hand_landmarks:
                if len(detection_info.multi_hand_landmarks) == 2:
                    mp_drawing.draw_landmarks(
                        image, detection_info.multi_hand_landmarks[0], mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(
                        image, detection_info.multi_hand_landmarks[1], mp_holistic.HAND_CONNECTIONS)
                else:
                    mp_drawing.draw_landmarks(
                        image, detection_info.multi_hand_landmarks[0], mp_holistic.HAND_CONNECTIONS)

            cv2.imshow('MediaPipe Holistic', image)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

    elif config.model == 'holistic':
        # drawing utils
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        # gesture recognition
        hand_dectector = SimpleGestureDetector()

        # For webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            # get hand dector images
            detection_info = hand_dectector.detectPose(image)


            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, detection_info.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, detection_info.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, detection_info.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, detection_info.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow('Gesture Recognition', image)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

    return

def test_pose_landmarks():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow('MediaPipe Holistic', image)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
    cap.release()

def test_run():
    # model = ...
    checkpoint = torch.load('Pretrained models/egogesture_resnext_101_RGB_32.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    output = model(torch.Tensor((28,28)))
    return

if __name__ == "__main__":
    #test_landmarks()
    #test_pose_landmarks()
    run()