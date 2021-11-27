from object_detection import app_object_detection
from utils import app_sendonly_video, app_sendonly_audio
from view_map import app_view_map

import logging
import threading
import streamlit as st

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

logger = logging.getLogger(__name__)


def main():
    st.header("OtanSpotter Prototype v1.0")

    video_sendonly_page = "Video Frames"
    view_map_page = "Location Tracking"
    object_detection_page = "Real-time Object Detection"

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            video_sendonly_page,
            object_detection_page,
            view_map_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == object_detection_page:
        app_object_detection()
    elif app_mode == video_sendonly_page:
        app_sendonly_video()
    elif app_mode == view_map_page:
        app_view_map()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
