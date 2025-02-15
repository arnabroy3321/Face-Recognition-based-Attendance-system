# Face Recognition Attendance System  

## 📌 Overview  
This is a **Face Recognition Attendance System** using **OpenCV** and **Local Binary Patterns Histograms (LBPH)**.  
The system allows users to:  
- **Register their face** (store images for training).  
- **Train the model** using stored images.  
- **Recognize users** and **mark attendance** in an Excel file (`attendance.xlsx`).  

## 🛠️ Dependencies  
Before running the project, install the required dependencies:  

### **1. Install Required Packages**  
Run the following command to install dependencies:  

pip install opencv-python pandas openpyxl numpy


Face-Recognition-Attendance/
│── registered_users/       # Folder storing user images  
│── trainer.yml             # Trained model file  
│── attendance.xlsx         # Attendance log file  
│── face_attendance.py      # Main Python script  
│── README.md               # Project documentation  


NOTE: Delete the attendance.xlsx or you can remove demo attendance