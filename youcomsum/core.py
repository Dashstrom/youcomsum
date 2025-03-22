"""Core module."""

import locale
import logging
import pathlib
import re
from collections import Counter
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import pycountry
from openai import OpenAI
from tqdm.auto import tqdm  # type: ignore[import-untyped]
from transformers import Pipeline, pipeline  # type: ignore[import-untyped]
from youtube_comment_downloader import (
    YoutubeCommentDownloader,
)

_downloader = YoutubeCommentDownloader()
_client = OpenAI()
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
DEFAULT_CONTEXT_SIZE = 16000
TAGS = {
    "Very Negative": 0,
    "Negative": 0.25,
    "Neutral": 0.5,
    "Positive": 0.75,
    "Very Positive": 1,
}


def get_video_id(text: str) -> str:
    """Get the ID of a video and validate it.

    Examples:
    - http://youtu.be/tVGH-g6OQhg
    - http://www.youtube.com/watch?v=tVGH-g6OQhg&feature=feed
    - http://www.youtube.com/embed/tVGH-g6OQhg
    - http://www.youtube.com/v/tVGH-g6OQhg?version=3&amp;hl=en_US
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


def fix_markdown(text: str, indent: int = 0) -> str:
    """Fix markdown."""
    min_header_level = (
        min((len(header) for header in RE_HEADER.findall(text)), default=0)
        - indent
    )

    def _replace_header(match: "re.Match[str]") -> str:
        level = len(match[2]) - min_header_level
        return f"{match[1]}{level * '#'} "

    text = RE_HEADER.sub(_replace_header, text)
    return text.strip()


_PIPE: Optional[Pipeline] = None


def summarize_youtube_comment(
    video: str,
    lang: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> str:
    """Generate report for a youtube video."""
    global _PIPE  # noqa: PLW0603
    if _PIPE is None:
        _PIPE = pipeline(
            "text-classification",
            model="tabularisai/multilingual-sentiment-analysis",
            max_length=512,
            truncation_strategy="only_first",
            truncation=True,
        )

    if lang is None:
        lang = get_default_lang()
    language = pycountry.languages.get(alpha_2=lang)

    video_id = get_video_id(video)
    logger.info("Summarize %s", video_id)
    logger.info("Fetching the video comments ...")
    text_size = -len(SEP)
    morsels = []
    batches = []
    comments = []
    counter = {tag: 0 for tag in TAGS}
    for comment in tqdm(
        _downloader.get_comments(video_id), desc="Download comments"
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

    results = _PIPE(comments)
    counter = Counter(result["label"] for result in results)
    rating = 0.0
    for value, count in counter.items():
        rating += TAGS[value] * count
    rating /= len(comments)

    logger.info("Sizes of batches: %s", [len(batch) for batch in batches])
    answers: list[str] = []
    for batch in tqdm(batches, desc="Process batches"):
        completion = _client.chat.completions.create(
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

    logger.info("Generate final answer ...")
    completion = _client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SUMMARIZE_PROMPT.replace(
                    "{LANGUAGE}", language.name
                ),
            },
            {"role": "user", "content": SEP.join(answers)},
        ],
    )
    result = completion.choices[0].message.content
    if result is None:
        err = "No message content generated"
        raise ValueError(err)

    logger.info("Generate rating answer ...")
    completion = _client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": RATING_PROMPT.replace("{LANGUAGE}", language.name),
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

    return fix_markdown(result) + "\n\n" + fix_markdown(rating_text)
