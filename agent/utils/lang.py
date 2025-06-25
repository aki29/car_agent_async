from __future__ import annotations
import re, time, threading, pathlib, urllib.request, tempfile, shutil, fasttext, os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

_RE_KANA = re.compile(r"[ぁ-んァ-ン]")
_RE_ZH = re.compile(r"[\u4e00-\u9fff]")
_MODEL_PATH = pathlib.Path(__file__).parent / "lid.176.ftz"

if not _MODEL_PATH.exists():
    print("[LangDetector] downloading lid.176.ftz …")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        shutil.move(tmp.name, _MODEL_PATH)
FT_MODEL = fasttext.load_model(str(_MODEL_PATH))

_state = threading.local()


def _init_state():
    _state.last_lang = os.getenv("DEFAULT_LANG", "en")
    _state.last_ts = 0.0


_init_state()

_TTL_SEC = 300
_SHORT_LEN = 3


class LangDetector:
    @staticmethod
    def _regex(text: str) -> str | None:
        if _RE_KANA.search(text):
            return "ja"
        return None

    def detect(self, text: str) -> str:
        text = text.strip()

        if len(text) <= _SHORT_LEN and time.time() - _state.last_ts < _TTL_SEC:
            return _state.last_lang

        lang = self._regex(text)
        if lang:
            _state.last_lang, _state.last_ts = lang, time.time()
            return lang

        lbl, _conf = FT_MODEL.predict(text, k=1)
        code = lbl[0].replace("__label__", "")

        if code.startswith("ja"):
            lang = "ja"
        elif code in ("zh", "zh-tw", "zh_Hant", "zh-TW"):
            lang = "zh-tw"
        elif code in ("zh-cn", "zh_Hans", "zh-CN"):
            lang = "zh-cn"
        else:
            lang = "en"

        _state.last_lang, _state.last_ts = lang, time.time()
        return lang
