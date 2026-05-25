import streamlit as st
import cv2
import numpy as np
from fer import FER
import pyttsx3
import random


# =========================================
# TEXT TO SPEECH ENGINE
# =========================================

tts_engine = pyttsx3.init()

tts_engine.setProperty('rate', 160)


# =========================================
# SPEAK FUNCTION
# =========================================

def speak_response(text):

    try:

        if tts_engine._inLoop:

            tts_engine.endLoop()

    except:

        pass

    tts_engine.say(text)

    tts_engine.runAndWait()


# =========================================
# FER EMOTION DETECTOR
# =========================================

detector = FER(mtcnn=True)


# =========================================
# STREAMLIT PAGE
# =========================================

st.set_page_config(

    page_title="AI Emotion Robot",

    page_icon="🤖"

)


st.title("🤖 AI Emotion Detection Robot")

st.subheader(

    "Capture Your Face and Detect Emotion Using AI"

)

st.write("System Running Successfully ✅")


# =========================================
# EMOJIS
# =========================================

emotion_emojis = {

    "happy": "😄",

    "sad": "😢",

    "angry": "😠",

    "neutral": "😐",

    "surprise": "😲",

    "fear": "😨",

    "disgust": "🤢"

}


# =========================================
# EMOTION RESPONSES
# =========================================

emotion_responses = {

    "happy": [

        "You look very happy today.",

        "Keep smiling and enjoy your day.",

        "Your happiness is inspiring.",

        "Stay positive and cheerful."

    ],

    "sad": [

        "You seem a little sad.",

        "Everything will become okay.",

        "Stay strong and calm.",

        "Better days are coming."

    ],

    "angry": [

        "You seem angry right now.",

        "Take a deep breath and relax.",

        "Stay calm and peaceful.",

        "Control your emotions carefully."

    ],

    "neutral": [

        "You look calm and focused.",

        "Your mood seems balanced.",

        "Stay productive and positive.",

        "You seem peaceful today."

    ],

    "surprise": [

        "Wow, you look surprised.",

        "That reaction seems unexpected.",

        "Hope it is good news.",

        "Life is full of surprises."

    ],

    "fear": [

        "You seem worried.",

        "Relax and trust yourself.",

        "Stay brave and confident.",

        "Take deep breaths slowly."

    ],

    "disgust": [

        "You seem uncomfortable.",

        "Take care of yourself.",

        "Stay relaxed and calm.",

        "Take a short break."

    ]
}


# =========================================
# CAMERA INPUT
# =========================================

picture = st.camera_input(

    "📸 Capture Your Face"

)


# =========================================
# PROCESS IMAGE
# =========================================

if picture is not None:


    # Convert image to OpenCV format

    file_bytes = np.asarray(

        bytearray(picture.read()),

        dtype=np.uint8

    )


    image = cv2.imdecode(

        file_bytes,

        cv2.IMREAD_COLOR

    )


    # Convert BGR to RGB

    rgb_image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )


    # Detect emotions

    results = detector.detect_emotions(

        rgb_image

    )


    # If face detected

    if results:


        result = results[0]

        emotions = result["emotions"]


        # Get highest confidence emotion

        emotion = max(

            emotions,

            key=emotions.get

        )


        confidence = emotions[emotion]


        # Face Box

        x, y, w, h = result["box"]


        cv2.rectangle(

            rgb_image,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )


        # Emotion label on image

        cv2.putText(

            rgb_image,

            emotion,

            (x, y - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2

        )


        # Show image

        st.image(

            rgb_image,

            channels="RGB"

        )


        # Emoji

        emoji = emotion_emojis.get(

            emotion,

            "🙂"

        )


        # Random response

        response = random.choice(

            emotion_responses.get(

                emotion,

                ["You are doing great."]
            )

        )


        # Show emotion

        st.success(

            f"{emoji} Detected Emotion: "
            f"{emotion} "
            f"({confidence:.2f})"

        )


        # Show response

        st.info(

            f"🤖 AI Assistant: {response}"

        )


        # Speak response

        speak_response(response)


    else:

        st.error(

            "❌ No face detected. Please try again."

        )
