import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO


class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the YOLOv10 model 
        self.model = YOLO(r'Model/test1.pt')  # Update this path to your model

        # Set up the main window
        self.setWindowTitle("ROC")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: black; color: white;")



        # Create a label to display the video feed or no connection
        self.display_label = QLabel(self)
        self.display_label.setGeometry(100, 30, 600, 500)
        self.display_label.setStyleSheet("border: 2px solid white;")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setText("no connection")
        self.display_label.setStyleSheet("font-size: 20px; color: white;")

        # Create a label to display the image above the "no connection" message
        self.image_label = QLabel(self)
        self.image_label.setGeometry(250, 100, 300, 200)
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('pyQT/drone.jpg')  # Update with your image path
        self.image_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))

        # Create a button to connect or disconnect the camera
        self.connect_button = QPushButton("Connect", self)
        self.connect_button.setGeometry(350, 520, 100, 30)
        self.connect_button.clicked.connect(self.handle_connect_button)

        # Create a button to start/stop recording (initially hidden)
        self.record_button = QPushButton("Start Recording", self)
        self.record_button.setGeometry(500, 520, 120, 30)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.hide()  # Hidden until camera is connected

        # Set up a label to display the recording time (hidden initially)
        self.timer_label = QLabel(self)
        self.timer_label.setGeometry(600, 20, 100, 50)
        self.timer_label.setStyleSheet("font-size: 20px; color: white;")
        self.timer_label.hide()

        # Set up a timer to update the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Timer for counting recording time
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)

        # Variable to hold the selected camera index
        self.cap = None
        self.selected_camera = None
        self.is_connected = False
        self.is_recording = False
        self.start_time = None
        self.out = None
        self.storage_location = None

    def handle_connect_button(self):
        """Handle the Connect/Disconnect button click."""
        if not self.is_connected:
            self.scan_cameras()
        else:
            self.disconnect_camera()

    def scan_cameras(self):
        """Scan and list available camera devices."""
        available_cameras = []
        for i in range(5):  # Adjust the range depending on the number of connected cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        if available_cameras:
            # Create a combo box to let the user select a camera
            self.camera_selector = QComboBox(self)
            self.camera_selector.addItems([f"Camera {i}" for i in available_cameras])

            selected_camera = QMessageBox.question(self, 'Select Camera', 'Select a camera:', QMessageBox.Ok)
            if selected_camera == QMessageBox.Ok:
                self.start_webcam(available_cameras[0])  # Assuming the user selects the first camera
        else:
            QMessageBox.warning(self, "No Cameras", "No available cameras were found.")

    def start_webcam(self, camera_index):
        """Start the selected webcam."""
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            self.is_connected = True
            self.connect_button.setText("Disconnect")
            self.record_button.show()  # Show the record button once the camera is connected
            self.record_button.setText("Start Recording")
            self.display_label.setText("")  # Clear the no connection label
            self.image_label.hide()  # Hide the image when the camera connects
            self.timer.start(20)  # Start updating the video feed

    def disconnect_camera(self):
        """Stop the camera and disconnect."""
        self.is_connected = False
        if self.cap is not None:
            self.cap.release()
        self.connect_button.setText("Connect")
        self.display_label.setText("no connection")
        self.image_label.show()  # Show the image again when disconnected
        self.record_button.hide()  # Hide the record button when disconnected
        self.timer.stop()
        if self.is_recording:
            self.stop_recording()

    def toggle_recording(self):
        """Start or stop recording the webcam feed."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording the webcam feed."""
        if not self.is_connected:
            return

        options = QFileDialog.Options()
        file, _ = QFileDialog.getSaveFileName(self, "Select File", "", "MP4 Video Files (*.mp4);;All Files (*)", options=options)

        if file:
            if not file.endswith('.mp4'):
                file += '.mp4'
            self.storage_location = file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(self.storage_location, fourcc, 20.0, (frame_width, frame_height))
            self.is_recording = True
            self.record_button.setText("Stop Recording")

            # Start recording time counter
            self.start_time = time.time()
            self.timer_label.show()  # Show the timer label
            self.recording_timer.start(1000)  # Update the timer every second

    def stop_recording(self):
        """Stop recording and reset the recording state."""
        self.is_recording = False
        if self.out is not None:
            self.out.release()
            self.out = None
        self.record_button.setText("Start Recording")
        self.recording_timer.stop()
        self.timer_label.hide()  # Hide the timer label
        self.timer_label.setText("")

    def update_recording_time(self):
        """Update the recording time counter every second."""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self.timer_label.setText(f"{minutes:02}:{seconds:02}")

    def update_frame(self):
        """Update the frame with the YOLOv8 model output."""
        ret, frame = self.cap.read()
        if ret:
            # Run YOLOv8 inference on the frame
            results = self.model(frame)
            result_frame = results[0].plot()

            # Convert the frame to QImage for display
            qt_img = self.convert_cv_qt(result_frame)
            self.display_label.setPixmap(qt_img)

            # If recording, save the frame
            if self.is_recording and self.out is not None:
                recorded_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                self.out.write(recorded_frame)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QImage."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(600, 400, Qt.KeepAspectRatio)
        return QPixmap.fromImage(convert_to_Qt_format)

    def closeEvent(self, event):
        """Handle closing the app."""
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
