import streamlit as st
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
import io
import time


def process(name, image, server_url: str):
    m = MultipartEncoder(fields={"file": (name, image, "image/jpg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


def upload():

    st.title("Video Live Detection YoloV5")

    input_video = st.file_uploader("insert video")

    if input_video:
        # video1=open("10 sec cli.mp4","rb")
        # st.video(video1)
        # st.video(open(input_video,"rb"))

        backend = 'https://www.youtube.com/watch?v=3SxlPAQbASE'
        segments = process("inference.mp4", input_video, backend)

        video1 = open("inference.mp4", "rb")
        st.video(video1)

        time.sleep(5)


if __name__ == "__main__":
    main()