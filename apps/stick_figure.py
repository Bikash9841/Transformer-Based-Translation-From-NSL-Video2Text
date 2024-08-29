
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# converts orginal video to stick figure representation
def conv_to_stick_fig(video_path, upload_dir):
    # For webcam input:
    cap = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

    fourcc = cv2.VideoWriter_fourcc(*'H264')

    videoWriter1 = cv2.VideoWriter(
        f"{upload_dir}/o.mp4", fourcc, 30.0, (224, 224))
    videoWriter2 = cv2.VideoWriter(
        f"{upload_dir}/c.mp4", fourcc, 30.0, (224, 224))

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                    # If loading a video, use 'break' instead of 'continue'.
                    # continue

                image = cv2.resize(image, (224, 224),
                                   interpolation=cv2.INTER_AREA)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                results_p = pose.process(image)

                img = np.zeros_like(image, dtype=np.uint8)

                mp_drawing.draw_landmarks(
                    img,
                    results_p.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(), hands_or_pose='Pose')

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                            hands_or_pose='Hands')

                # Flip the image horizontally for a selfie-view display.
                image = cv2.flip(image, 1)
                img = cv2.flip(img, 1)

                # save the mediapipe hand and pose video
                videoWriter1.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                videoWriter2.write(img)

                # cv2.imshow('MediaPipe Hand and Pose', img)
                # cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
    cap.release()
    videoWriter1.release()
    videoWriter2.release()
    return f"{upload_dir}/o.mp4", f"{upload_dir}/c.mp4"
# cv2.destroyAllWindows()
