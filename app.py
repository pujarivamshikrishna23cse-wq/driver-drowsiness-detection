import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.title("Driver Drowsiness Detection")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Example text
        cv2.putText(
            img,
            "Camera Working",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="driver-monitor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)