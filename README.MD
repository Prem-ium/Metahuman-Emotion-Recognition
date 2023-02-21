# Emotion Detection for Metahuman

Capstone project exploring emotion mimicking abilities of metahumans. 

Group Members: Prem Patel (Prem-ium), Gabe Vindas (GabeV95), Matthew , & Dustin

### Installation
Run locally:
1. Clone this repository, cd into it, and install dependancies:
```sh
   git clone https://github.com/Prem-ium/EmotionDetection.git
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