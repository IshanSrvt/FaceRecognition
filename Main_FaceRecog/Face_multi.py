import face_recognition
import cv2
import numpy as np

def LoadFaces():
    Ishan_image = face_recognition.load_image_file("Ishan.jpg")
    Ishan_face_encoding = face_recognition.face_encodings(Ishan_image)[0]

    Manish_image = face_recognition.load_image_file("Manish Srivastava.png")
    Manish_face_encoding = face_recognition.face_encodings(Manish_image)[0]

    Shirish_image = face_recognition.load_image_file("Shirish Srivastava.png")
    Shirish_face_encoding = face_recognition.face_encodings(Shirish_image)[0]

    Shikha_image = face_recognition.load_image_file("Shikha Srivastava.png")
    Shikha_face_encoding = face_recognition.face_encodings(Shikha_image)[0]

    Shivansh_image = face_recognition.load_image_file("Shivansh.jpg")
    Shivansh_face_encoding = face_recognition.face_encodings(Shivansh_image)[0]

    Devansh_image = face_recognition.load_image_file("Devansh.jpg")
    Devansh_face_encoding = face_recognition.face_encodings(Devansh_image)[0]

    known_face_encodings = [
        Ishan_face_encoding,
        Manish_face_encoding,
        Shirish_face_encoding,
        Shikha_face_encoding,
        Shivansh_face_encoding,
        Devansh_face_encoding
        ]
    known_face_names = [
        "Ishan",
        "Manish",
        "Shirish",
        "Shikha",
        "Shivansh",
        "Devansh"
    ]

    return known_face_encodings, known_face_names;

video_capture = cv2.VideoCapture(0-0)
known_face_encodings, known_face_names = LoadFaces()

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

   # face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    #for face_landmarks in face_landmarks_list:

   #     for facial_feature in face_landmarks.keys():
    #        pts = np.array([face_landmarks[facial_feature]], np.int32)
    #        pts = pts.reshape((-1,1,2))
    #        cv2.polylines(frame, [pts], False, (0,255,0))

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 4)

            cv2.rectangle(frame, (left, bottom - 31), (right, bottom), (255, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()