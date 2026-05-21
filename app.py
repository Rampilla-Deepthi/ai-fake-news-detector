import streamlit as st

import numpy as np

from newspaper import Article

from transformers import pipeline

from sumy.parsers.plaintext import PlaintextParser

from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.lsa import LsaSummarizer

import easyocr

from PIL import Image

from youtube_transcript_api import YouTubeTranscriptApi
import nltk

nltk.download('punkt')

nltk.download('punkt_tab')

def get_youtube_transcript(video_id):

    transcript = YouTubeTranscriptApi().fetch(video_id)

    text = ""

    for item in transcript:

        text += item.text + " "

    return text



st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 AI Fake News Detection System")

st.markdown("""
Analyze:
- 📰 News Articles
- ▶️ YouTube Videos
- 🖼️ News Images/Screenshots

using AI-powered misinformation detection.
""")

st.sidebar.title("About Project")

st.sidebar.info("""
This project uses:
- NLP
- OCR
- AI Text Classification
- YouTube Transcript Analysis
- Source Credibility Logic
""")

st.write("AI-powered misinformation detection project")



@st.cache_resource
def load_model():

    classifier = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )

    return classifier



classifier = load_model()

reader = easyocr.Reader(['en'])

def summarize_text(text):

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    summarizer = LsaSummarizer()

    summary = summarizer(parser.document, 3)

    final_text = ""

    for sentence in summary:
        final_text += str(sentence) + " "

    return final_text



tab1, tab2, tab3 = st.tabs([
    "📰 Article",
    "▶️ YouTube",
    "🖼️ Image"
])



with tab1:

    article_url = st.text_input(
        "Enter News Article URL"
    )



with tab2:

    youtube_url = st.text_input(
        "Enter YouTube Video URL"
    )



with tab3:

    uploaded_image = st.file_uploader(
        "Upload News Image",
        type=["jpg", "jpeg", "png"]
    )

if st.button("Analyze"):

    with st.spinner("Analyzing content with AI..."):

        try:

            if uploaded_image is not None:

                st.write("Extracting text from image...")

                image = Image.open(uploaded_image)

                results = reader.readtext(
                    np.array(image)
                )

                text = ""

                for item in results:
                    text += item[1] + " "




            elif youtube_url:

                st.write("Extracting YouTube transcript...")

                if "v=" in youtube_url:

                    video_id = youtube_url.split("v=")[1][:11]

                else:

                    video_id = youtube_url.split("/")[1][:11]

                try:

                    text = get_youtube_transcript(video_id)

                except:

                    st.error(
                        "Could not retrieve YouTube transcript.\n\n"
                        "This may happen because YouTube blocks "
                        "cloud server requests."
                    )

                    st.stop()



            else:

                st.write("Extracting article...")

                article = Article(article_url)

                article.download()

                article.parse()

                text = article.text



            st.write("Summarizing content...")

            short_text = summarize_text(text)



            st.write("Analyzing with AI...")

            result = classifier(short_text)



            label = result[0]['label']

            score = result[0]['score']



            st.markdown("---")

            st.subheader("📊 Analysis Result")



            trusted_sources = [
                "bbc.com",
                "reuters.com",
                "apnews.com",
                "ndtv.com",
                "thehindu.com",
                "indianexpress.com"
            ]



            trusted = False



            for source in trusted_sources:

                if source in article_url or source in youtube_url:

                    trusted = True

                    break

            if uploaded_image is not None:

                if label == "LABEL_1":

                    st.warning(
                        "Image text may contain sensational "
                        "or unclear OCR patterns.\n\n"
                        "Manual verification recommended."
                    )

                else:

                    st.success("Possibly Real Image Content")



            elif youtube_url:

                if trusted and label == "LABEL_1":

                    st.warning(
                        "Trusted source video detected.\n\n"
                        "AI detected conversational or "
                        "sensational language patterns, "
                        "but content may still be genuine."
                    )

                elif label == "LABEL_1":

                    st.warning(
                        "Video contains suspicious or "
                        "emotion-heavy language.\n\n"
                        "Further verification recommended."
                    )

                else:

                    st.success("Possibly Real Video Content")



            else:

                if trusted and label == "LABEL_1":

                    st.warning(
                        "Trusted source detected.\n\n"
                        "AI found sensational language patterns, "
                        "but article may still be genuine."
                    )

                elif label == "LABEL_1" and score > 0.95:

                    st.error("Likely Fake or Misleading News")

                else:

                    st.success("Likely Real News")



            st.write(f"Confidence Score: {score:.2f}")
            st.subheader("🔎 Verification Resources")

            st.markdown("""
            Check these trusted fact-checking websites:

            - [Snopes](https://www.snopes.com)
            - [Reuters Fact Check](https://www.reuters.com/fact-check)
            - [Alt News](https://www.altnews.in)
            - [BoomLive](https://www.boomlive.in)
            - [FactCheck.org](https://www.factcheck.org)
            """)

            st.markdown("---")

            st.subheader("📝 Content Summary")

            st.info(short_text)



        except Exception as e:

            st.error(f"Error: {e}")
st.markdown("---")

st.caption(
    "AI Fake News Detection Project "
    "using NLP, OCR, and Multimedia Analysis"
)