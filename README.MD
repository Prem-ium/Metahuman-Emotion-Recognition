# Emotion Detection for Metahuman

![image](https://user-images.githubusercontent.com/80719066/229880499-9b973ded-d2ea-453f-8590-f873696ad039.png)
(Image taken by @GabeV95)
### Installation
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
- Dustin Lynn
