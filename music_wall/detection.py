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



def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def detect_squares(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 100
    max_area = 1500
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+h]
            cv2.imwrite(os.path.abspath(f'../wall-music/resources/ROI_{image_number}.png'), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.imwrite(os.path.abspath(f'../wall-music/resources/ROI2_{image_number}.png'), image)
            image_number += 1

    #cv2.imshow('sharpen', sharpen)
    #cv2.imshow('close', close)
    #cv2.imshow('thresh', thresh)
    cv2.imshow('image', image)

    return image_number

def bound_detect(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(os.path.abspath(f'../wall-music/resources/Agray.png'), blurred)

    edged = auto_canny(blurred)
    cv2.imwrite(os.path.abspath(f'../wall-music/resources/AAsomeedgdes.png'), edged)

    # detect lines in the image using hough lines technique
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 60, np.array([]), 50, 5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 5)



    edges = cv2.Canny(image, 10, 200)
    cv2.imwrite(os.path.abspath(f'../wall-music/resources/AAedges.png'), edges)

    #thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imwrite(os.path.abspath(f'../wall-music/resources/Athres.png'), thresh)


    #blur = cv2.medianBlur(gray, 5)
    #sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    min_area = 50
    max_area = 3000000

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = original[y:y + h, x:x + w]
            cv2.imwrite(os.path.abspath(f'../wall-music/resources/test/NEW_{ROI_number}.png'), ROI)
            ROI_number += 1
    return ROI_number

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


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

def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # get image details
    img = cv2.imread(path)
    h, w = img.shape[:2]

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

        # fetch normalized coordinates of bounds
        x = object_.bounding_poly.normalized_vertices[0].x
        x1 = object_.bounding_poly.normalized_vertices[2].x
        y = object_.bounding_poly.normalized_vertices[0].y
        y1 = object_.bounding_poly.normalized_vertices[2].y

        # calculate coordinates realtive to image size
        x = int(x*w)
        x1 = int(x1*w)
        y = int(y*h)
        y1 = int(y1*h)
        plt.imshow(img[y:y1, x:x1])
        plt.show()
        #ROI = original[y:y + h, x:x + w]



def detect_writing(path):
    # Loads the image into memory
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    # perform api request
    image = vision.Image(content=content)
    # gets writing from the request
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    outputs = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))
                    outputs.append(word_text)

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
    return outputs
