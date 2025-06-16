Folder Structure
aadhaar_yolo/
├── aadhaar_ocr_pipeline.py       # Main script
├── best.pt                       # Trained YOLOv8 model
├── test/                         # Input test images
├── annotated_test/               # Annotated output images
├── not_found_test/               # Failed OCR crops
└── aadhaar_predictions_test.csv  # Results CSV

Requirements:
Python 3.8+
Tesseract OCR installed
Dependencies: pip install opencv-python pytesseract ultralytics

How to Run:

Create required folders:
mkdir test annotated_test not_found_test

Place your images in the test/ folder
Run the pipeline:
python aadhaar_ocr_pipeline.py
