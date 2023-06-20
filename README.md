### Project for license plate detection and recognition.

To run the project you need:
1. Download weights  from https://drive.google.com/drive/folders/1kz-_czvaMhZ0ubTsiPN2fOthPXyZzCGe?usp=sharing
2. Put them in the folder ./weights
3. run ```pip install -r requirements.txt```
4. run ```python main.py``` with folowing arguments:
   ```--img_path``` - for input image (e.g. ```-img_path /home/ivan/cars.jpeg```)
   ```--out_path``` - for output json with OCR of license plates (e.g. ```--out_path temp_results```)
