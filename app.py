import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("Driver Drowsiness Detection")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.putText(
        img,
        "Camera Running",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="driver-monitor",
    video_frame_callback=video_frame_callback
)