import cv2
import pytesseract
import re
import os
import csv
from ultralytics import YOLO

# --- Configs ---
MODEL_PATH = r"C:\Users\hp\aadhar_yolo\best.pt"
TESSERACT_PATH = r'E:\ocr\tesseract.exe'
TEST_DIR = r"C:\Users\hp\aadhar_yolo\aadhaar_yolo\test"
ANNOTATED_DIR = r"C:\Users\hp\aadhar_yolo\aadhaar_yolo\annotated_test"
NOT_FOUND_DIR = r"C:\Users\hp\aadhar_yolo\aadhaar_yolo\not_found_test"
OUTPUT_CSV = r"C:\Users\hp\aadhar_yolo\aadhaar_yolo\aadhaar_predictions_test.csv"


# Control output verbosity (set to False for less console output)
VERBOSE = True

# --- Setup ---
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(NOT_FOUND_DIR, exist_ok=True)

def extract_aadhaar_from_image(image_path, model):
    image = cv2.imread(image_path)
    results = model(image)[0]
    annotated_image = image.copy()
    aadhaar_number = "not found"
    best_confidence = 0.0
    
    # Check if we have any detections
    if results.boxes is None or len(results.boxes) == 0:
        if VERBOSE:
            print(f"No detections for {os.path.basename(image_path)}")
        save_path = os.path.join(ANNOTATED_DIR, os.path.basename(image_path))
        cv2.imwrite(save_path, annotated_image)
        return aadhaar_number, 0.0

    if VERBOSE:
        print(f"Found {len(results.boxes)} detection(s) for {os.path.basename(image_path)}")
    
    # Process each detection
    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        confidence = results.boxes.conf[i].item()
        
        if VERBOSE:
            print(f"  Detection {i+1}: Box ({x1},{y1})-({x2},{y2}), Confidence: {confidence:.3f}")
        
        # Add padding and ensure bounds
        h, w = image.shape[:2]
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Extract crop
        cropped = image[y1:y2, x1:x2]
        
        # Draw bounding box on annotated image
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f'{confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OCR with multiple PSM modes
        ocr_configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
        ]
        
        best_result = ""
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(cropped, config=config).strip()
                # Clean the text - remove all non-digits
                clean_text = re.sub(r'\D', '', text)
                
                if VERBOSE:
                    print(f"    OCR attempt: '{text}' -> cleaned: '{clean_text}'")
                
                # Check if we got exactly 12 digits
                if len(clean_text) == 12 and clean_text.isdigit():
                    best_result = clean_text
                    if VERBOSE:
                        print(f"    ✓ Found valid Aadhaar: {best_result}")
                    break
                    
            except Exception as e:
                if VERBOSE:
                    print(f"    OCR error: {e}")
                continue
        
        # If we found a valid Aadhaar number, use it
        if best_result:
            aadhaar_number = best_result
            best_confidence = confidence
            break
        else:
            # Save failed crop for debugging
            crop_path = os.path.join(NOT_FOUND_DIR, f"{os.path.basename(image_path)}_crop_{i+1}.jpg")
            cv2.imwrite(crop_path, cropped)
            if VERBOSE:
                print(f"    ✗ No valid Aadhaar found, saved crop: {crop_path}")

    # Save annotated image
    save_path = os.path.join(ANNOTATED_DIR, os.path.basename(image_path))
    cv2.imwrite(save_path, annotated_image)
    
    if VERBOSE:
        print(f"Final result for {os.path.basename(image_path)}: {aadhaar_number}")
    return aadhaar_number, best_confidence

# --- Main Processing Loop ---
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    # Get ALL images in the directory
    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process")
    print(f"Output will be saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")
    
    # Statistics tracking
    successful_extractions = 0
    failed_extractions = 0
    detection_failures = 0
    ocr_failures = 0
    
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "aadhaar_number", "confidence", "status"])

        for idx, image_file in enumerate(image_files, 1):
            if VERBOSE:
                print(f"\n[{idx}/{total_images}] Processing: {image_file}")
                print('-' * 50)
            else:
                # Simple progress indicator for non-verbose mode
                if idx % 50 == 0 or idx == 1 or idx == total_images:
                    print(f"Processing... {idx}/{total_images}")
            
            try:
                image_path = os.path.join(TEST_DIR, image_file)
                aadhaar_number, confidence = extract_aadhaar_from_image(image_path, model)
                
                # Determine status and update statistics
                if aadhaar_number != "not found":
                    successful_extractions += 1
                    status = "SUCCESS"
                    aadhaar_number_excel = f'="{aadhaar_number}"'
                else:
                    failed_extractions += 1
                    status = "FAILED"
                    aadhaar_number_excel = "not found"

                writer.writerow([image_file, aadhaar_number_excel, f"{confidence:.3f}", status])
                
                if VERBOSE:
                    print(f"RESULT: {aadhaar_number} (Confidence: {confidence:.3f}, {status})")
                
                # Progress update every 10 images (verbose) or 100 images (non-verbose)
                progress_interval = 10 if VERBOSE else 100
                if idx % progress_interval == 0 or idx == total_images:
                    success_rate = (successful_extractions / idx) * 100
                    print(f"Progress: {idx}/{total_images} ({success_rate:.1f}% success rate so far)")
                    
            except Exception as e:
                print(f"ERROR processing {image_file}: {e}")
                writer.writerow([image_file, "ERROR", "N/A", f"ERROR: {str(e)}"])
                failed_extractions += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Overall success rate: {(successful_extractions/total_images)*100:.1f}%")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Annotated images saved to: {ANNOTATED_DIR}")
    print(f"Failed crops saved to: {NOT_FOUND_DIR}")