import cv2
import torch
from ultralytics import YOLO
import time
import os

class YOLO_RaspberryPi:
    def __init__(self, model_path='yolov8n.pt'):
        print("Initializing YOLO on Raspberry Pi 5...")
        
        try:
            self.model = YOLO(model_path)  
            print("Model loaded successfully")
            
            # Optimize model for Raspberry Pi
            self.model.amp = False           
            self.model.fuse()      
            
            # Test model with a simple inference
            try:
                with torch.no_grad():
                    test_tensor = torch.zeros(1, 3, 320, 320)  # Smaller test size
                    test_result = self.model(test_tensor)
                print("Model inference test passed")
            except Exception as e:
                print(f"Model inference test failed: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  # Re-raise so the script stops if model fails to load
        
        # Get device info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Torch threads: {torch.get_num_threads()}")
        
        # Optimize torch for Raspberry Pi
        torch.set_num_threads(4)  # Limit threads to avoid overloading Pi

    def process_usb_camera(self, camera_index=0):
        """Process video from USB camera"""
        print(f"Initializing USB camera (index {camera_index})...")
        
        # Initialize camera with optimized settings for Raspberry Pi
        cap = cv2.VideoCapture(camera_index)
        time.sleep(1.0)  # Give camera time to initialize
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Could not open USB camera at index {camera_index}")
            print("Available camera indices: 0, 1, 2... Try different indices if needed.")
            return
            
        # Test camera
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            cap.release()
            return
            
        print("USB camera initialized successfully")
        print(f"Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
        print("YOLO on Raspberry Pi 5 - Processing USB Camera...")
        print("Press 'q' to quit, 'p' to pause, 's' to save current frame")
        print("'[' to decrease text size, ']' to increase text size")
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        pause = False
        
        # Customization parameters
        text_scale = 0.5  # Smaller text (default: 0.5x size)
        
        # Reduce processing resolution for better performance
        processing_size = 320
        
        while True:
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break
                    
                original_frame = frame.copy()
                
                # Resize for processing
                height, width = frame.shape[:2]
                scale = processing_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
                
                try:
                    # Run inference with optimized settings
                    results = self.model(
                        frame_resized,
                        imgsz=processing_size,
                        verbose=False,
                        conf=0.5,
                        iou=0.5,
                        max_det=20,
                        half=False,
                        device='cpu'
                    )
                    
                    # Print detection info occasionally
                    if frame_count % 30 == 0:
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            print(f"Frame {frame_count}: Detected {len(boxes)} objects")
                            for i, box in enumerate(boxes[:3]):
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = self.model.names[cls]
                                print(f" {class_name}: {conf:.2f}")
                        else:
                            print(f"Frame {frame_count}: No detections")
                    
                    # Get annotated frame
                    if len(results) > 0:
                        annotated = results[0].plot()
                        annotated = cv2.resize(annotated, (width, height))
                    else:
                        annotated = original_frame
                        
                except Exception as e:
                    print(f"Inference error: {e}")
                    annotated = original_frame
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - start_time)
                    start_time = time.time()
                    print(f"Current FPS: {fps:.1f}")
                
                # Add info overlays
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Frame: {frame_count}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Text: {text_scale:.1f}x', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    cv2.putText(annotated, f'Detections: {len(results[0].boxes)}',
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(annotated, 'Detections: 0',
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('YOLO-Raspberry Pi 5 - USB Camera', annotated)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                pause = not pause
                print("Paused" if pause else "Resumed")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Frame saved as {filename}")
            elif key == ord(']'):
                text_scale = min(1.5, text_scale + 0.1)
                print(f"Text scale increased to: {text_scale:.1f}")
            elif key == ord('['):
                text_scale = max(0.3, text_scale - 0.1)
                print(f"Text scale decreased to: {text_scale:.1f}")
            elif key == ord('d'):
                print(f"Model: {self.model}")
                print(f"Model classes: {self.model.names}")
                
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    # Initialize YOLO
    yolo = YOLO_RaspberryPi('yolov8n.pt')
    
    # Process USB camera
    yolo.process_usb_camera(0)