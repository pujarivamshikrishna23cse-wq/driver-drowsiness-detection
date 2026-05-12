import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

st.title("Driver Drowsiness Detection")

mp_face_mesh = mp.solutions.face_mesh


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            cv2.putText(
                img,
                "Driver Awake",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                "No Face Detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="driver-monitor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)