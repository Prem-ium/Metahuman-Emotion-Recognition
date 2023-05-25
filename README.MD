<h1 align="center"> 🧠💻 Real-Time Emotion Recognition Metahuman🧠💻 </h1>
<p align="center">Creating <i>real-time</i> facial and emotion recgonition software to pair with Unreal Engine 5 Metahuman to predict and mimic user emotions.</p>
<p align="right"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/><img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/><img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/><img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/></p>
Our objective is to develop a 3D Virtual Human (3DVH) prototype that can mirror a user’s emotional reaction. The user emotional reaction is captured using a computer vison-based technique and a camera system.  The VH will mimic user emotion in real-time.

## Supported Emotions:
- Anger 😡
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprised 😲

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
Configure your variables in a .env file within the same directory. All .env variables are optional and have default values if not specified. 

`HEADLESS`=True or False. Whether to open a GUI for testing webcam accuracy. Defaults to True.

`PRODUCTION`= True or False. Whether the program is running in Unreal Engine or not.

`DELAY`= Integer value of how many seconds the program will wait before starting the next iteration

`FILE_PATH`= Path of directory containing the model and weights

`WEIGHTS`=Name of eight Model being used.

## Donations
I've been working on this project for a few months now, and I'm really happy with how it's turned out. It's also been a helpful tool for users to detect a user's age, gender, and emotions with a hyper-realistic virtual human built using Unreal Engine 5. I'm currently working on adding new features to the script and working on other similar programs to generate passive income. I'm also working on making the script more user-friendly and accessible to a wider audience.

I'm accepting donations through BuyMeCoffee or GitHub Sponsors. Any amount you can donate will be greatly appreciated.

<div style="display: grid; grid-template-columns: 1fr 1fr; justify-content: center; gap: 10px;">
    <b>Buy Me a Coffee:</b>
    <a href="https://www.buymeacoffee.com/prem.ium" target="_blank" style="text-decoration: none;">
            <img src="https://raw.githubusercontent.com/Prem-ium/youtube-analytics-bot/main/output-examples/media/coffee-logo.png" alt="Buy Me A Coffee" style="height: 80px; width: 320px; box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5);"></a>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; justify-content: center; gap: 10px;">
    <b>GitHub Sponsors:</b>
        <a href="https://github.com/sponsors/Prem-ium" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA" alt="GitHub Sponsor" style="height: 80px; width: 320px; box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5);">
        </a>
</div>

Your donations will help me to cover the costs of hosting the project, developing new features, and marketing the project to a wider audience.
Thank you for your support!

# Capstone Group Members
- Python Scripting: Prem Patel ([@Prem-ium](https://github.com/Prem-ium))
- Integrating Script/Blueprint into Unreal Engine: Gabe Vindas ([@GabeV95](https://github.com/GabeV95)) & Dustin Lynn ([@Onemorehell](https://github.com/Onemorehell))
- Creating Unreal Engine Emotion Animations: Matthew Goetz
