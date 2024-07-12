<h1 align="center">🧠💻 Real-Time Emotion Recognition Metahuman 🧠💻</h1>

<p align="center">Creating <i>real-time</i> facial and emotion recognition software to pair with Unreal Engine 5 Metahuman to predict and mimic user emotions.</p>

<p align="right">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/>
  <a href="https://github.com/sponsors/Prem-ium" target="_blank">
    <img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AA" alt="Github Sponsor"/>
  </a>
</p>

<p>Our objective is to develop a 3D Virtual Human (3DVH) prototype that can mirror a user’s emotional reaction. The user emotional reaction is captured using a computer vision-based technique and a camera system. The virtual human will mimic user emotions in real-time.</p>


# Contributors 🎓

- **Python Scripting:** Prem Patel ([@Prem-ium](https://github.com/Prem-ium))
- **Unreal Engine Integration:** Gabe Vindas ([@GabeV95](https://github.com/GabeV95)) & Dustin Lynn ([@Onemorehell](https://github.com/Onemorehell))
- **Emotion Animations in Unreal Engine:** Matthew Goetz


## Supported Emotions 😃:
- Anger 😡
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprised 😲

## Our Virtual Humans 🎭
![image](https://user-images.githubusercontent.com/80719066/229881331-faa04b08-8524-4901-9398-e8b927542c98.png)
They're terrifying, we know.

## Example Results 🎯
![image](https://user-images.githubusercontent.com/80719066/229880499-9b973ded-d2ea-453f-8590-f873696ad039.png)
(Showcasing Happy & Surprised)

## Features 🔑

- Emotion Recognition
- Age & Gender Prediction
- Enviornmental Variables Customization
- Headless mode
- Flexible Emotion Selection: Choose between the most common emotion every X captured emotions or the latest emotion.

## Installation Process & Setup Process ⚙️️ 
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

### Enviornmental Variables 🛠️
Configure your variables in a .env file within the same directory. All .env variables are optional and have default values if not specified. 

| Variable     | Description                                                        | Default Value |
|--------------|--------------------------------------------------------------------|---------------|
| HEADLESS     | True or False. Whether to open a GUI for testing webcam accuracy.   | True          |
| PRODUCTION   | True or False. Whether the program is running in Unreal Engine.     | False         |
| DELAY        | Integer value of how many seconds the program will wait before starting the next iteration | - |
| FILE_PATH    | Path of directory containing the model and weights                  | - |
| WEIGHTS      | Name of the model being used                                        | - |

## Need Help? 🤔
I regret to inform you that I'm currently unable to respond to any issues for this repository. I have disabled the ability to start an issue in this repository. Please refrain from contacting me regarding any issues at this time.

## Donations 🎁
<a href="https://github.com/sponsors/Prem-ium" target="_blank">
        <img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA" alt="GitHub Sponsor" img width="15%">
</a>
<a href="https://www.buymeacoffee.com/prem.ium" target="_blank">
        <img src="https://raw.githubusercontent.com/Prem-ium/youtube-analytics-bot/main/output-examples/media/coffee-logo.png" alt="Buy Me A Coffee" img width="15%">
</a>

## License

[MIT](https://choosealicense.com/licenses/mit/)


