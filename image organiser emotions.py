import cv2
from fer import FER  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os
from collections import defaultdict

def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to read {image_path}")
        return None, None

    # Detect emotions
    emotions = emotion_detector.detect_emotions(frame)
    
    if not emotions:
        return None, None
    
    # Process detected faces
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        emotion_label = max(emotion["emotions"], key=emotion["emotions"].get)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put emotion label above the rectangle
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36, 255, 12), 2)
    
    return frame, emotion_label

def batch_image_emotion_recognition(image_paths):
    emotion_groups = defaultdict(list)
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            frame, emotion_label = process_image(image_path)
            if frame is not None and emotion_label is not None:
                emotion_groups[emotion_label].append((frame, os.path.basename(image_path)))
        else:
            print(f"Invalid file path: {image_path}")
    
    # Display images grouped by emotion
    for emotion, images in emotion_groups.items():
        num_images = len(images)
        cols = min(3, num_images)  # Max 3 images per row
        rows = (num_images + cols - 1) // cols  # Calculate rows dynamically
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]  # Make it 2D for consistency
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, (frame, filename) in enumerate(images):
            row, col = divmod(idx, cols)
            axes[row][col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[row][col].axis('off')
            axes[row][col].set_title(filename)
        
        plt.suptitle(f"Emotion: {emotion}", fontsize=16)
        plt.tight_layout()
        plt.show()

# Initialize the FER detector
emotion_detector = FER(mtcnn=True)

def main():
    image_paths = input("Enter image file paths separated by commas: ").split(',')
    image_paths = [path.strip() for path in image_paths]
    batch_image_emotion_recognition(image_paths)

if __name__ == "__main__":
    main()
