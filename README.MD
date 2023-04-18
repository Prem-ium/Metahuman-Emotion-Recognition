# Emotion Recognition Capable Metahumans
Our objective is to develop a 3D Virtual Human (3DVH) prototype that can mirror a user’s emotional reaction. The user emotional reaction is captured using a computer vison-based technique and a camera system.  The VH will mimic user emotion in real-time.

## Supported Emotions:
- Anger
- Fear
- Happy
- Neutral
- Sad
- Surprised

## Our Virtual Humans
![image](https://user-images.githubusercontent.com/80719066/229881331-faa04b08-8524-4901-9398-e8b927542c98.png)
## Example Results 
![image](https://user-images.githubusercontent.com/80719066/229880499-9b973ded-d2ea-453f-8590-f873696ad039.png)
(Showcasing Happy & Surprised)

### Installation Process & Setup Process
Run locally:
1. Clone this repository, cd into it, and install dependancies:
```sh
   git clone https://github.com/Prem-ium/Metahuman-Emotion-Recognition.git
   cd EmotionDetection
   pip install -r requirements.txt
   ```
2. Configure your `.env` file (See below and example for options)
3. Run the main script:
```sh
   python emotional-detection-main.py
```
4. Open Unreal Engine Project & Run the Blueprint
5. Click the button to trigger the text reader to process the most common emotion recorded.
6. The Metahuman mimics the user's most common emotion.
7. Repeat Steps 5-6 until desired termination

### Enviornmental Variables
Configure your variables in a .env file within the same directory.

`HEADLESS`=True or False. Whether to open a GUI for testing webcam accuracy. Defaults to True.

`PRODUCTION`= True or False. Whether the program is running in Unreal Engine or not.

`DELAY`= Integer value of how many seconds the program will wait before starting the next iteration

`FILE_PATH`= Path of directory containing the model and weights

`WEIGHTS`=Name of eight Model being used.

# Group Members
- Prem Patel ([@Prem-ium](https://github.com/Prem-ium))
- Gabe Vindas ([@GabeV95](https://github.com/GabeV95))
- Matthew Goetz
- Dustin Lynn ([@Onemorehell](https://github.com/Onemorehell))
