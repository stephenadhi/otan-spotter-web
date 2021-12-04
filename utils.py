import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import queue
import streamlit as st
import logging
import urllib.request
from typing import Literal
from pathlib import Path
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    RTCConfiguration
)
import streamlit.components.v1 as components
# Video getting and saving
import urllib
import m3u8
import time
import streamlit as st
import pafy  # needs youtube_dl

logger = logging.getLogger(__name__)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
def home_page_html():
    HtmlFile = open("home.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=1600, width=1600)



def app_sendonly_video():

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    playing = st.checkbox("Playing", value=False)
    frame_rate = 30
    webrtc_ctx = webrtc_streamer(
        key="sendonly-video", desired_playing_state=playing,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,
        video_html_attrs={
            "style": {"width": "120%", "margin": "0 auto", "border": "5px black solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )

    st.write(f"The frame rate is set as {frame_rate}. Video style is changed.")


def app_sendonly_audio():
    """A sample to use WebRTC in sendonly mode to transfer audio frames
    from the browser to the server and visualize them with matplotlib
    and `st.pyplot`."""
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True},
    )

    fig_place = st.empty()

    fig, [ax_time, ax_freq] = plt.subplots(
        2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    )

    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
                sound_window_buffer = sound_window_buffer.set_channels(
                    1
                )  # Stereo to mono
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break

def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def watch_video(video_url, image_placeholder, n_segments=1,
                n_frames_per_segment=60):
    """Gets a video clip,
        video_url: YouTube video URL
				image_placeholder: streamlit image placeholder <where clip will display
				n_segments: how many segments of the YouTube stream to pull (max 7)
				n_frames_per_segment: how many frames each segment should be, max ~110
				"""
    ####################################
    # SETUP BEFORE PLAYING VIDEO TO GET VIDEO URL
    ####################################

    # Speed up playing by only showing every nth frame
    skip_n_frames = 5
    video_warning = st.warning("showing a clip from youTube...")

    # Use pafy to get the 360p url
    url = video_url
    video = pafy.new(url)

    best = video.getbest(preftype="mp4")  #  Get best resolution stream available
    # In this specific case, the 3rd entry is the 360p stream,
    #   But that's not ALWAYS true.
    #medVid = video.streams[2]
    medVid = best
    #  load a list of current segments for live stream
    playlist = m3u8.load(medVid.url)

    # will hold all frames at the end
    # can be memory intestive, so be careful here
    frame_array = []

    # Speed processing by skipping n frames, so we need to keep track
    frame_num = 0

    ###################################
    # Loop over each frame of video, and each segment
    ###################################
    #  Loop through all segments
    for i in playlist.segments[0:n_segments]:

        capture = cv2.VideoCapture(i.uri)

        # go through every frame in each segment
        for i in range(n_frames_per_segment):

            # open CV function to pull a frame out of video
            success, frame = capture.read()
            if not success:
                break

            # Skip every nth frame to speed processing up
            if (frame_num % skip_n_frames != 0):
                frame_num += 1
                pass
            else:
                frame_num += 1

                # Show the image in streamlit, then pause
                image_placeholder.image(frame, channels="BGR")
                time.sleep(0.5)

    # Clean up everything when finished
    capture.release()  # free the video

    # Removes the "howing a clip from youTube" message"
    video_warning.empty()

    st.write("Done with clip, frame length", frame_num)

    return None


def camera_view():
    # streamlit placeholder for image/video
    image_placeholder = st.empty()

    # url for video
    # Jackson hole town square, live stream
    video_url = "https://www.youtube.com/watch?v=DoUOrTJbIu4"

    # Description for the sidebar options
    st.sidebar.write("Set options for processing video, then process a clip.")

    # Make a slider bar from 60-360 with a step size of 60
    #  st.sidebar is going to make this appear on the sidebar
    n_frames = 30  # Frames per 'segment"
    n_segments = st.sidebar.slider("How many frames should this video be:",
                                   n_frames, n_frames * 6, n_frames, step=n_frames, key="spots",
                                   help="It comes in 7 segments, 100 frames each")

    # We actually need to know the number of segments,
    #   so convert the total number of frames to the number of segments we want
    n_segments = int(n_segments / n_frames)

    # Add a "go" button to show the clip in streamlit
    if st.sidebar.button("watch video clip"):
        watch_video(video_url=video_url,
                    image_placeholder=image_placeholder,
                    n_segments=n_segments,
                    n_frames_per_segment=n_frames)