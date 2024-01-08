import math
import mediapipe as mp
import cv2


class SimpleGestureDetector:
    # region: Member variables
    # mediaPipe configuration hands object
    __mpHands = mp.solutions.hands
    __mpPose = mp.solutions.holistic

    # mediaPipe detector objet
    __mpHandDetector = None
    __mpPoseDetector = None

    def __init__(self):
        self.__setDefaultHandConfiguration()
        self.__setDefaultPoseConfiguration()

    def __setDefaultHandConfiguration(self):
        self.__mpHandDetector = self.__mpHands.Hands(
            # default = 2
            max_num_hands=2,
            # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully (default= 0.5)
            min_detection_confidence=0.5,
            # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. (default = 0.5)
            min_tracking_confidence=0.5
        )

    def __setDefaultPoseConfiguration(self):
        self.__mpPoseDetector = self.__mpPose.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def __getEuclideanDistance(self, posA, posB):
        return math.sqrt((posA.x - posB.x) ** 2 + (posA.y - posB.y) ** 2)

    def __isThumbNearIndexFinger(self, thumbPos, indexPos):
        return self.__getEuclideanDistance(thumbPos, indexPos) < 0.1

    def detectHands(self, capture):
        if self.__mpHandDetector is None:
            return

        #image = capture.color
        # Input image must contain three channel rgb data.
        image = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
        # lock image for hand detection
        image.flags.writeable = False
        # start handDetector on current image
        detectorResults = self.__mpHandDetector.process(image)
        # unlock image
        image.flags.writeable = True

        if detectorResults.multi_hand_landmarks:
            for landmarks in detectorResults.multi_hand_landmarks:
                self.simpleGesture(landmarks)

        return detectorResults

    def detectPose(self, capture):
        if self.__mpHandDetector is None:
            return

        #image = capture.color
        # Input image must contain three channel rgb data.
        image = capture#image = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
        # lock image for hand detection
        image.flags.writeable = False
        # start handDetector on current image
        detectorResults = self.__mpPoseDetector.process(image)
        # unlock image
        image.flags.writeable = True

        if detectorResults.left_hand_landmarks:
            self.simpleGesture(detectorResults.left_hand_landmarks)

        return detectorResults

    def simpleGesture(self, handLandmarks):

        thumbIsOpen = False
        indexIsOpen = False
        middelIsOpen = False
        ringIsOpen = False
        pinkyIsOpen = False

        pseudoFixKeyPoint = handLandmarks.landmark[2].x
        if handLandmarks.landmark[3].x < pseudoFixKeyPoint and handLandmarks.landmark[4].x < pseudoFixKeyPoint:
            thumbIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[6].y
        if handLandmarks.landmark[7].y < pseudoFixKeyPoint and handLandmarks.landmark[8].y < pseudoFixKeyPoint:
            indexIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[10].y
        if handLandmarks.landmark[11].y < pseudoFixKeyPoint and handLandmarks.landmark[12].y < pseudoFixKeyPoint:
            middelIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[14].y
        if handLandmarks.landmark[15].y < pseudoFixKeyPoint and handLandmarks.landmark[16].y < pseudoFixKeyPoint:
            ringIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[18].y
        if handLandmarks.landmark[19].y < pseudoFixKeyPoint and handLandmarks.landmark[20].y < pseudoFixKeyPoint:
            pinkyIsOpen = True

        if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FIVE!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FOUR!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
            print("THREE!")

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("TWO!")

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("ONE!")

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("ROCK!")

        elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("SPIDERMAN!")

        elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("FIST!")

        elif not indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen and self.__isThumbNearIndexFinger(
                handLandmarks.landmark[4], handLandmarks.landmark[8]):
            print("OK!")
        elif not thumbIsOpen and not indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print('FUCK U TOO')

        print("FingerState: thumbIsOpen? " + str(thumbIsOpen) + " - indexIsOpen? " + str(
            indexIsOpen) + " - middelIsOpen? " +
              str(middelIsOpen) + " - ringIsOpen? " + str(ringIsOpen) + " - pinkyIsOpen? " + str(pinkyIsOpen))

        return