import cv2
from ultralytics import YOLO

# 1. Load your OBB model weights
model_path = r"C:\Users\rishi\OneDrive\Desktop\IP-Tactile\Finger_Exploration_APP\output\runs\run_20260526_131359\models\training_run\weights\best.pt"
model = YOLO(model_path)

# 2. Path to your test video file (Replace with your actual video filename)
video_path = "video.mp4" 

# Open a handle to the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print(f"--- Running video inference using {model_path} ---")
print("Press 'q' on your keyboard while the window is active to quit early.")

# 3. Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    
    # If the video ends or breaks, exit the loop
    if not success:
        print("End of video file or failed to read frame.")
        break
        
    # Run YOLO prediction on the current frame
    # stream=True optimizes memory utilization for long videos
    results = model.predict(source=frame, conf=0.25, stream=False, verbose=False)
    
    # Plot the predicted oriented bounding boxes onto the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame in a desktop window
    cv2.imshow("YOLO Live Video Detection", annotated_frame)
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Inference stopped early by user.")
        break

# 4. Clean up and close windows properly
cap.release()
cv2.destroyAllWindows()