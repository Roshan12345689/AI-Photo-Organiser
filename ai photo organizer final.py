import cv2
import face_recognition
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from fer import FER  # type: ignore

def organize_by_faces(image_paths, clf):
    processed_images = defaultdict(list)
    unknown_faces = []
    face_distance_threshold = 0.5
    fixed_height = 500

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            continue

        test_image = face_recognition.load_image_file(image_path)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            
            try:
                predicted_label = clf.predict([face_encoding])[0]
                known_encodings = np.array(clf.support_vectors_)
                similarity = cosine_similarity([face_encoding], known_encodings).max()

                if similarity > face_distance_threshold:
                    name = predicted_label
            except Exception as e:
                print("[ERROR] Prediction failed:", e)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
            cv2.putText(rgb_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            
            aspect_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            new_width = int(fixed_height * aspect_ratio)
            resized_image = cv2.resize(rgb_image, (new_width, fixed_height), interpolation=cv2.INTER_AREA)

            if name == "Unknown":
                unknown_faces.append(resized_image)
            else:
                processed_images[name].append(resized_image)
    
    display_grouped_images(processed_images, unknown_faces)

def organize_by_emotions(image_paths):
    
    emotion_detector = FER(mtcnn=True)
    emotion_groups = defaultdict(list)
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Invalid file path: {image_path}")
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to read {image_path}")
            continue

        emotions = emotion_detector.detect_emotions(frame)
        if not emotions:
            continue

        for emotion in emotions:
            (x, y, w, h) = emotion["box"]
            emotion_label = max(emotion["emotions"], key=emotion["emotions"].get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 12), 4)
            emotion_groups[emotion_label].append(frame)
    
    display_grouped_emotions(emotion_groups)

def display_grouped_images(processed_images, unknown_faces):
    final_output = []
    max_row_width = max([sum(img.shape[1] for img in images) for images in processed_images.values()] + 
                        [sum(img.shape[1] for img in unknown_faces)] + [0])
    
    for person, images in processed_images.items():
        person_group = np.hstack(images)
        pad_width = max_row_width - person_group.shape[1]
        person_group = cv2.copyMakeBorder(person_group, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_output.append(person_group)
    
    if unknown_faces:
        unknown_group = np.hstack(unknown_faces)
        pad_width = max_row_width - unknown_group.shape[1]
        unknown_group = cv2.copyMakeBorder(unknown_group, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_output.append(unknown_group)
    
    if final_output:
        final_image = np.vstack(final_output)
        cv2.imshow("Grouped Face Recognition Output", cv2.resize(final_image, (1200, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def display_grouped_emotions(emotion_groups):
    fixed_height = 300  # Ensuring consistent image height
    final_output = []
    max_row_width = max([sum(img.shape[1] for img in images) for images in emotion_groups.values()] + [0])
    
    for emotion, images in emotion_groups.items():
        resized_images = [cv2.resize(img, (int(fixed_height * img.shape[1] / img.shape[0]), fixed_height)) for img in images]
        emotion_group = np.hstack(resized_images)
        pad_width = max_row_width - emotion_group.shape[1]
        emotion_group = cv2.copyMakeBorder(emotion_group, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_output.append(emotion_group)

    if final_output:
        final_image = np.vstack(final_output)
        cv2.imshow("Grouped Emotion Recognition Output", cv2.resize(final_image, (1200, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    choice = input("Choose organization method: 1 - By Faces, 2 - By Emotions: ")
    image_paths = input("Enter image file paths separated by commas: ").split(',')
    image_paths = [path.strip() for path in image_paths]
    
    if choice == '1':
        with open("face_recognition_model.pkl", "rb") as f:
            clf = pickle.load(f)
        organize_by_faces(image_paths, clf)
    elif choice == '2':
        organize_by_emotions(image_paths)
    else:
        print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()
