import datetime, os, traceback

# Description: Reads the last line of a text file and converts it to an integer to pass to unreal engine.

# Replace the file path with the path to your 'common_emotions.txt' text file path, created using emotional-detection-main.py when in production mode.
txtFilePath = 'C:/Users/Frost/Desktop/CodingProjects/EmotionDetection-main/common_emotions.txt'

global output
emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
with open(txtFilePath, 'r') as file:
    lines = file.readlines()
    lines.reverse()
    for line in lines:
        if line.strip():
            try:
                output = int(line)
                print(f"Common Index: {output}\t\tEmotion: {emotion_classes[output]}\t{datetime.datetime.now()}\n")
                break
            except ValueError:
                print(traceback.format_exc())