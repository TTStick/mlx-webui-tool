"""把上传的文件转成可放进上下文的纯文本。"""
from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

from .config import MAX_ATTACHMENT_CHARS

_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp", ".tiff"}
_MEDIA = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv"}
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class AttachmentError(ValueError):
    pass


def _extension(name: str) -> str:
    match = re.search(r"\.[A-Za-z0-9]+$", name)
    return match.group(0).lower() if match else ""


def _decode(data: bytes) -> str:
    if data[:4096].count(b"\0") > 0 and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise AttachmentError("这看起来是二进制文件，目前只支持文本、代码、PDF 和 Word 文档")
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise AttachmentError("这个 PDF 有密码保护，无法读取")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise AttachmentError(f"PDF 解析失败：{exc}") from exc
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise AttachmentError("没有从这个 PDF 中提取到文字，它可能是扫描件")
    return text


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AttachmentError("Word 文档解析失败，只支持 .docx 格式") from exc
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(f"{_WORD_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_WORD_NS}t"))
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_text(name: str, data: bytes) -> tuple[str, bool]:
    """返回 (文本, 是否被截断)。"""
    extension = _extension(name)
    if extension in _IMAGE:
        raise AttachmentError("当前使用的是文本模型，暂不支持图片")
    if extension in _MEDIA:
        raise AttachmentError("暂不支持音频和视频文件")
    if extension == ".pdf":
        text = _pdf_text(data)
    elif extension == ".docx":
        text = _docx_text(data)
    else:
        text = _decode(data)
    truncated = len(text) > MAX_ATTACHMENT_CHARS
    return text[:MAX_ATTACHMENT_CHARS], truncated
