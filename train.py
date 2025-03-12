import sys
from contextlib import redirect_stdout, redirect_stderr
from ultralytics import YOLO
from multiprocessing import freeze_support

def train_model():
    # Load a YOLOv8 model
    model = YOLO(r'yolov10n.pt')

    # Train the model
    model.tune(data=r'C:\Users\tusha\OneDrive\Documents\Projects\NasscomProject\CCTV Surveillance Dataset\data.yaml', epochs=25, imgsz=320, device=0)

def main():
    train_model()

if __name__ == '__main__':
    freeze_support()
    
    # Path to the file where output will be saved
    output_file = 'training_output.txt'

    # Open the file in write mode and redirect stdout and stderr to this file
    with open(output_file, 'w') as f:
        with redirect_stdout(f), redirect_stderr(f):
            main()