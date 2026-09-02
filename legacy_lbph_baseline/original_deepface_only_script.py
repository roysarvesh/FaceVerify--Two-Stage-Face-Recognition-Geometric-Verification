import os
import cv2
from deepface import DeepFace

DATABASE = r"C:\Users\dronr\Desktop\Complete-OpenCV-with-Projects-main\opencv_project\Faces\train"
IMAGE = r"C:\Users\dronr\Desktop\Complete-OpenCV-with-Projects-main\opencv_project\Faces\train\Ben Afflek\1.jpg"

best_person = None
best_distance = 999

for person in os.listdir(DATABASE):

    person_folder = os.path.join(DATABASE, person)

    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):

        train_image = os.path.join(person_folder, file)

        try:

            result = DeepFace.verify(
                img1_path=IMAGE,
                img2_path=train_image,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=False
            )

            if result["distance"] < best_distance:

                best_distance = result["distance"]
                best_person = person

        except:
            pass

print("\nBest Match:", best_person)
print("Distance:", best_distance)

img = cv2.imread(IMAGE)

cv2.putText(
    img,
    best_person,
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imshow("Recognition", img)

cv2.waitKey(0)
cv2.destroyAllWindows()