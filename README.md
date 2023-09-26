### Project for license plate detection and recognition.
**The task was to implement the detection and recognition of car license plates within 2 hours**

To run the project you need:
1. Download weights  from https://drive.google.com/drive/folders/1kz-_czvaMhZ0ubTsiPN2fOthPXyZzCGe?usp=sharing
2. Put them in the folder ./weights
3. run ```pip install -r requirements.txt```
4. run ```python main.py``` with folowing arguments:
   ```--img_path``` - for input image (e.g. ```-img_path /home/ivan/cars.jpeg```)
   ```--out_path``` - for output json with OCR of license plates (e.g. ```--out_path temp_results```)


Plates detection is implemented with yolov5 model. OCR model was taken from https://github.com/sirius-ai/LPRNet_Pytorch and doesnt work properly at the time.

Results:

![image](https://github.com/egorovivannn/dummy_license_plate/assets/88214807/4f7e7ff6-472d-40fa-9a61-328dbff24205)

![image](https://github.com/egorovivannn/dummy_license_plate/assets/88214807/52be44b5-cc66-4286-8215-a5be569dec4b)
