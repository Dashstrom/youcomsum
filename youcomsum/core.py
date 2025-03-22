"""Core module."""

import locale
import logging
import pathlib
import re
from collections import Counter
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import pycountry
from tqdm.auto import tqdm  # type: ignore[import-untyped]

RE_USER = re.compile(r"\xa0@[\w\d_\-]+\xa0")
logger = logging.getLogger(__name__)
PROMPTS = pathlib.Path(__file__).parent / "prompts"
BATCH_PROMPT = (PROMPTS / "batch_prompt.txt").read_text("utf-8")
SUMMARIZE_PROMPT = (PROMPTS / "summarize_prompt.txt").read_text("utf-8")
RATING_PROMPT = (PROMPTS / "rating_prompt.txt").read_text("utf-8")
RATING_TEMPLATE = (PROMPTS / "rating_template.txt").read_text("utf-8")
RE_VIDEO_ID = re.compile(r"[0-9a-zA-Z\-_]{3,}")
SEP = "\n#######################################\n"
RE_HEADER = re.compile(r"(^|\n)(#+) ")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONTEXT_SIZE = 25000
TAGS = {
    "Very Negative": 0,
    "Negative": 0.25,
    "Neutral": 0.5,
    "Positive": 0.75,
    "Very Positive": 1,
}


def get_default_lang() -> Any:
    """Get default system language."""
    locale_str = locale.getdefaultlocale()[0]
    if locale_str:
        _, country_code = locale_str.split("_")
        if country_code:
            return country_code
    try:
        import ctypes

        windll = ctypes.windll.kernel32
        locale_str = locale.windows_locale[windll.GetUserDefaultUILanguage()]
        _, country_code = locale_str.split(".")[0].split("_")
        if country_code:
            return country_code
    except ImportError:
        pass
    return "en"


def fix_markdown(text: str, header: int = 1) -> str:
    """Fix markdown."""
    min_header_level = (
        min((len(match[2]) for match in RE_HEADER.finditer(text)), default=0)
        - header
    )

    def _replace_header(match: "re.Match[str]") -> str:
        level = len(match[2]) - min_header_level
        return f"{match[1]}{level * '#'} "

    text = RE_HEADER.sub(_replace_header, text)
    return text.strip()


def get_video_id(text: str) -> str:
    """Get the ID of a video and validate it.

    Examples:
    - http://youtu.be/dQw4w9WgXcQ
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=feed
    - http://www.youtube.com/embed/dQw4w9WgXcQ
    - http://www.youtube.com/v/dQw4w9WgXcQ?version=3&amp;hl=en_US
    """
    url_data = urlparse(text)
    if url_data.hostname == "youtu.be":
        return url_data.path[1:]
    if url_data.hostname in ("www.youtube.com", "youtube.com"):
        if url_data.path == "/watch":
            query = parse_qs(url_data.query)
            return str(query["v"][0])
        if url_data.path[:7] == "/embed/":
            return url_data.path.split("/")[2]
        if url_data.path[:3] == "/v/":
            return url_data.path.split("/")[2]
    return text


class YouComSum:
    def __init__(self) -> None:
        """Init."""
        logger.info("Loading modules and models...")
        from openai import OpenAI
        from transformers import pipeline  # type: ignore[import-untyped]
        from youtube_comment_downloader import YoutubeCommentDownloader

        self.downloader = YoutubeCommentDownloader()
        self.client = OpenAI()
        self.pipe = pipeline(
            "text-classification",
            model="tabularisai/multilingual-sentiment-analysis",
            max_length=512,
            truncation_strategy="only_first",
            truncation=True,
        )
        logger.info("Loading done !")

    def summarize(
        self,
        video: str,
        lang: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> str:
        """Generate report for a youtube video."""
        # REsolve the id of the video
        video_id = get_video_id(video)
        logger.info("Summarize %s", video_id)

        # Detect the system language
        if lang is None:
            lang = get_default_lang()
        language = pycountry.languages.get(alpha_2=lang)
        logger.info("The summary will be in %r", language.name)

        logger.info("Fetching the video comments ...")
        text_size = -len(SEP)
        morsels = []
        batches = []
        comments = []
        counter = {tag: 0 for tag in TAGS}
        for comment in tqdm(
            self.downloader.get_comments(video_id),
            desc="Download comments",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            text: str = comment["text"].replace("\r\n", "\n")
            text = RE_USER.sub("@USER", text)
            comments.append(text)
            text_size += len(text)
            text_size += len(SEP)
            morsels.append(text)
            if text_size > context_size:
                batches.append(SEP.join(morsels[:-1]))
                morsels = morsels[-1:]
                text_size = len(morsels[0])
        batches.append(SEP.join(morsels))

        # Process comments as batch
        logger.info("Sizes of batches: %s", [len(batch) for batch in batches])
        answers: list[str] = []
        for batch in tqdm(
            batches,
            desc="Process batches",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": BATCH_PROMPT.replace(
                            "{LANGUAGE}", language.name
                        ),
                    },
                    {"role": "user", "content": batch},
                ],
            )
            if completion.choices[0].message.content is None:
                err = "No message content generated"
                raise ValueError(err)
            answers.append(completion.choices[0].message.content)

        final_prompt = SEP.join(answers)
        logger.info(
            "Generate final summary (~%s tokens) ...",
            len(final_prompt.split()),
        )
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SUMMARIZE_PROMPT.replace(
                        "{LANGUAGE}", language.name
                    ),
                },
                {"role": "user", "content": final_prompt},
            ],
        )
        result = completion.choices[0].message.content
        if result is None:
            err = "No message content generated"
            raise ValueError(err)

        # Use sentiment analysis model
        results = self.pipe(comments)

        # Normalize result and generate rating
        counter = Counter(result["label"] for result in results)
        rating = 0.0
        for value, count in counter.items():
            rating += TAGS[value] * count
        rating /= len(comments)

        # Generate the final rating answer with rating
        logger.info("Generate rating answer ...")
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": RATING_PROMPT.replace(
                        "{LANGUAGE}", language.name
                    ),
                },
                {
                    "role": "user",
                    "content": RATING_TEMPLATE.replace(
                        "{RATING}", f"{rating * 4 + 1:.2f}"
                    )
                    .replace(
                        "{VERY_NEGATIVE}", str(counter.get("Very Negative", 0))
                    )
                    .replace("{NEGATIVE}", str(counter.get("Negative", 0)))
                    .replace("{NEUTRAL}", str(counter.get("Neutral", 0)))
                    .replace("{POSITIVE}", str(counter.get("Positive", 0)))
                    .replace(
                        "{VERY_POSITIVE}", str(counter.get("Very Positive", 0))
                    ),
                },
            ],
        )
        rating_text = completion.choices[0].message.content
        if rating_text is None:
            err = "No message content generated"
            raise ValueError(err)

        # Fix different header level
        return (
            fix_markdown(result, header=1)
            + "\n\n"
            + fix_markdown(rating_text, header=2)
        )
