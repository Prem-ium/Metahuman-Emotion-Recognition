# Name: Prem Patel (Prem-ium)
# Group 2: Metahuman
# Purpose: Create a program that can detect emotions from a webcam feed and display them to be mimicked by a Metahuman.
# Group Members: Gabe Vindas (GabeV95), Matthew Goetz, Dustin Lynn

import os, sys, traceback, argparse, dlib, cv2, datetime

from os                             import listdir
from os.path                        import isfile, join
from contextlib                     import contextmanager

from pathlib                        import Path
from wide_resnet                    import WideResNet
from collections                    import Counter

import numpy                        as np

from keras.utils.data_utils         import get_file
from keras.models                   import load_model
from keras_preprocessing.image      import img_to_array
from time                           import sleep
from dotenv                         import load_dotenv

# Load .env file
load_dotenv()

HEADLESS = os.environ.get("HEADLESS", "True")
if HEADLESS.lower() == "true":
    print("Running in headless mode")
    HEADLESS = True
else:
    print("Running in non-headless mode")
    HEADLESS = False

DELAY = int(os.environ.get("DELAY", "2"))

PRODUCTION = os.environ.get("PRODUCTION", "False")

if PRODUCTION.lower() == "true":
    print("Running in production mode")
    PRODUCTION = True
else:
    PRODUCTION = False

if not os.environ["FILE_PATH"]:
    modelPath = 'emotion_little_vgg_2.h5'
    weightsPath = 'weights.28-3.73.hdf5'
else:
    F_PATH = os.environ.get("FILE_PATH", "C:/Program Files/Epic Games/UE_5.1/Engine/Content/Python/")
    modelPath = F_PATH + 'emotion_little_vgg_2.h5'
    weightsPath = F_PATH + 'weights.28-3.73.hdf5'

if not PRODUCTION:
    print("Using model: " + modelPath)
    print("Using weights: " + weightsPath)

def cached_emotions_init(file="emotions.txt"):
    if not os.path.isfile(file):
        open(file, "a").close()
        print(f"Created new {file} file \n{datetime.datetime.now()}\n")
        print()
    else:
        print(f"{file} file already exists \n{datetime.datetime.now()}\n")
        print()

def append_cached_emotions(emotion_id, file="emotions.txt"):
    with open({file}, "a") as f:
        f.write(f"{emotion_id}\n")
        print(f"Appended {emotion_id} to {file} \n{datetime.datetime.now()}\n")

def close_cached_emotions(file="emotions.txt"):
    print(f"Closing {file} \n{datetime.datetime.now()}\n")
    with open(file, "a") as f:
        f.close()
    print(f'Deleting {file}\t{datetime.datetime.now()}')
    os.remove(file)

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def main():
    modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
    emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                       3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

    # Define our model parameters
    depth = 16
    k = 8
    weight_file = None
    margin = 0.4
    image_dir = None

    classifier = load_model(modelPath)
    pretrained_model = "https://github.com/Prem-ium/EmotionDetection/releases/download/Model/weights.28-3.73.hdf5"

    # Get our weight file
    if not weight_file:
        weight_file = get_file(weightsPath, pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)
    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    detector = dlib.get_frontal_face_detector()

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    emo_labels = []
    while len(emo_labels) < 10:
        ret, frame = cap.read()

        preprocessed_faces_emo = []

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        detected = detector(frame, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        preprocessed_faces_emo = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                    1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)
                break
                

            # make a prediction for Age and Gender
            results = model.predict(np.array(faces), verbose=0)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            for i, d in enumerate(detected):
                sleep(DELAY)
                preds = classifier.predict(preprocessed_faces_emo[i], verbose=0)[0]
                emo_labels.append(emotion_classes[preds.argmax()])
                if HEADLESS and not PRODUCTION:
                    age = int(predicted_ages[i])
                    gender = "F" if predicted_genders[i][0] > 0.4 else "M"
                    emotion = emo_labels[i]

                    print(f'{gender} {age}: {emotion}')
                append_cached_emotions(emo_labels[i])
                break

            if not HEADLESS:
                for i, d in enumerate(detected):
                    label = "{}, {}, {}".format(int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.4 else "M", emo_labels[i])
                    draw_label(frame, (d.left(), d.top()), label)
                    print(emo_labels[i])
                    break

            if len(emo_labels) >= 10:
                most_common_emo = Counter(emo_labels[-10:]).most_common(1)[0][0]
                most_common_emo_index = list(emotion_classes.values()).index(most_common_emo)
                print(f'Most common emotion: {most_common_emo} (Index to send to Unreal: {most_common_emo_index})')
                append_cached_emotions(most_common_emo_index, "common_emotions.txt")
                emo_labels = []

        if not HEADLESS:
            cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cap.release()

    if not HEADLESS:
        cv2.destroyAllWindows()
    close_cached_emotions()
    close_cached_emotions("common_emotions.txt")


if __name__ == '__main__':
    try:
        cached_emotions_init()
        cached_emotions_init("common_emotions.txt")
        main()
    except:
        print(traceback.format_exc())
