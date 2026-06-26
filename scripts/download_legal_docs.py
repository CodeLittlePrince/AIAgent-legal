#!/usr/bin/env python3
"""Download full Chinese legal texts into profile/legal/.

Priority:
1. Domestic: https://flk.npc.gov.cn (国家法律法规数据库)
2. Fallback: GitHub mirrors via HTTP proxy (if configured)
"""

from __future__ import annotations

import io
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "profile" / "legal"

FLK_BASE = "https://flk.npc.gov.cn"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 120.0

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零\d]+章")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零\d]+节")
ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百零\d]+条)(.*)$")


@dataclass(frozen=True)
class LawSpec:
    title: str
    filename: str
    mirror_url: str


LAWS: tuple[LawSpec, ...] = (
    LawSpec(
        title="中华人民共和国民法典",
        filename="中华人民共和国民法典.md",
        mirror_url="https://raw.githubusercontent.com/risshun/Chinese_Laws/master/基本法律/中华人民共和国民法典.md",
    ),
    LawSpec(
        title="中华人民共和国劳动法",
        filename="中华人民共和国劳动法.md",
        mirror_url="https://raw.githubusercontent.com/risshun/Chinese_Laws/master/非基本法律/中华人民共和国劳动法.md",
    ),
    LawSpec(
        title="中华人民共和国劳动合同法",
        filename="中华人民共和国劳动合同法.md",
        mirror_url="https://raw.githubusercontent.com/risshun/Chinese_Laws/master/非基本法律/中华人民共和国劳动合同法.md",
    ),
    LawSpec(
        title="中华人民共和国消费者权益保护法",
        filename="中华人民共和国消费者权益保护法.md",
        mirror_url="https://raw.githubusercontent.com/LawRefBook/Laws/master/民法商法/消费者权益保护法(2013-10-25).md",
    ),
    LawSpec(
        title="中华人民共和国刑法",
        filename="中华人民共和国刑法.md",
        mirror_url="https://raw.githubusercontent.com/risshun/Chinese_Laws/master/刑法/中华人民共和国刑法.md",
    ),
)


def _proxy_mounts() -> dict[str, httpx.HTTPTransport] | None:
    """Use HTTP proxy only (not SOCKS) for foreign mirror fallback."""
    proxy = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    if not proxy or proxy.startswith("socks"):
        return None
    transport = httpx.HTTPTransport(proxy=proxy)
    return {"http://": transport, "https://": transport}


def _flk_headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Referer": f"{FLK_BASE}/",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
    }


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _search_flk(client: httpx.Client, title: str) -> str | None:
    payload = {
        "searchRange": 1,
        "searchType": 1,
        "searchContent": title,
        "pageNum": 1,
        "pageSize": 10,
        "sxrq": [],
        "gbrq": [],
        "sxx": [],
        "gbrqYear": [],
        "flfgCodeId": [],
        "zdjgCodeId": [],
    }
    response = client.post(f"{FLK_BASE}/law-search/search/list", json=payload)
    response.raise_for_status()
    rows = response.json().get("rows", [])
    for row in rows:
        row_title = _strip_html(row.get("title", ""))
        if row_title == title:
            return row.get("bbbs")
    return None


def _download_flk_docx(client: httpx.Client, bbbs: str) -> bytes:
    response = client.get(
        f"{FLK_BASE}/law-search/download/pc",
        params={"bbbs": bbbs, "format": "docx"},
    )
    response.raise_for_status()
    payload = response.json()
    url = payload["data"]["url"]
    file_response = client.get(url)
    file_response.raise_for_status()
    return file_response.content


def _docx_to_markdown(docx_bytes: bytes, title: str) -> str:
    from docx import Document

    doc = Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text.strip().replace("\u3000", " ") for p in doc.paragraphs if p.text.strip()]

    lines: list[str] = [f"# {title}", ""]
    meta_written = False
    in_toc = False
    body_started = False

    for para in paragraphs:
        if para == title:
            continue
        normalized = re.sub(r"\s+", "", para)
        if normalized == "目录":
            in_toc = True
            continue

        if ARTICLE_RE.match(para):
            body_started = True
            in_toc = False

        if in_toc and not body_started:
            continue

        if not meta_written and para.startswith("（") and "通过" in para:
            lines.extend([f"> {para}", ""])
            meta_written = True
            continue

        if CHAPTER_RE.match(para):
            lines.extend([f"## {para}", ""])
            continue
        if SECTION_RE.match(para):
            lines.extend([f"### {para}", ""])
            continue

        article_match = ARTICLE_RE.match(para)
        if article_match:
            article_no, rest = article_match.groups()
            rest = rest.strip()
            if rest:
                lines.extend([f"#### {article_no}", "", rest, ""])
            else:
                lines.extend([f"#### {article_no}", ""])
            continue

        lines.extend([para, ""])

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _fetch_from_flk(client: httpx.Client, law: LawSpec) -> str | None:
    client.get(f"{FLK_BASE}/")
    bbbs = _search_flk(client, law.title)
    if not bbbs:
        print(f"  flk: no exact match for {law.title}", file=sys.stderr)
        return None
    print(f"  flk: found bbbs={bbbs}")
    docx_bytes = _download_flk_docx(client, bbbs)
    print(f"  flk: downloaded docx ({len(docx_bytes)} bytes)")
    markdown = _docx_to_markdown(docx_bytes, law.title)
    article_count = len(re.findall(r"^#### 第.+?条", markdown, re.MULTILINE))
    print(f"  flk: converted to markdown ({len(markdown)} chars, {article_count} articles)")
    if article_count < 10:
        print("  flk: too few articles, treating as failure", file=sys.stderr)
        return None
    return markdown


def _fetch_from_mirror(law: LawSpec) -> str | None:
    headers = {"User-Agent": USER_AGENT}
    mounts = _proxy_mounts()
    label = "mirror+proxy" if mounts else "mirror"
    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers=headers,
            follow_redirects=True,
            mounts=mounts,
        ) as client:
            response = client.get(law.mirror_url)
            response.raise_for_status()
            text = response.text.strip()
            if len(text) < 1000:
                return None
            if not text.startswith("# "):
                text = f"# {law.title}\n\n{text}"
            print(f"  {label}: fetched {len(text)} chars from {law.mirror_url}")
            return text if not text.endswith("\n") else text
    except httpx.HTTPError as exc:
        print(f"  {label}: failed {exc}", file=sys.stderr)
        return None


def _write_readme(output_dir: Path, results: list[tuple[str, str, int]]) -> None:
    lines = [
        "# 法律文档库（完整版）",
        "",
        "本目录存放用于 RAG 检索的**完整**中国法律 Markdown 文本。",
        "",
        "## 免责声明",
        "",
        "1. 文本来源于国家法律法规数据库（flk.npc.gov.cn）或公开镜像，仅供参考。",
        "2. 正式法律适用请以全国人大常务委员会公报及官方公布文本为准。",
        "3. AI 助手基于本文档生成的回答不构成法律意见。",
        "",
        f"**最近更新：** {date.today().isoformat()}",
        "",
        "## 文档列表",
        "",
        "| 文件 | 来源 | 大小 |",
        "|------|------|------|",
    ]
    for filename, source, size in results:
        lines.append(f"| `{filename}` | {source} | {size:,} bytes |")
    lines.extend(
        [
            "",
            "## 更新方式",
            "",
            "```bash",
            "python scripts/download_legal_docs.py",
            "```",
            "",
            "国内优先从 flk.npc.gov.cn 下载；失败时使用 GitHub 镜像（可配置 HTTP 代理）。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def download_all(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy excerpt files
    for old in output_dir.glob("*_节选.md"):
        old.unlink()
        print(f"Removed legacy excerpt: {old.name}")

    written: list[Path] = []
    readme_rows: list[tuple[str, str, int]] = []

    with httpx.Client(timeout=REQUEST_TIMEOUT, headers=_flk_headers(), verify=False) as flk_client:
        for law in LAWS:
            print(f"Processing {law.title} ...")
            content: str | None = None
            source = "unknown"

            try:
                content = _fetch_from_flk(flk_client, law)
                if content:
                    source = "flk.npc.gov.cn"
            except httpx.HTTPError as exc:
                print(f"  flk: error {exc}", file=sys.stderr)

            if content is None:
                content = _fetch_from_mirror(law)
                if content:
                    source = "github-mirror"

            if content is None:
                print(f"ERROR: failed to fetch {law.title}", file=sys.stderr)
                continue

            out_path = output_dir / law.filename
            out_path.write_text(content, encoding="utf-8")
            written.append(out_path)
            readme_rows.append((law.filename, source, out_path.stat().st_size))
            print(f"  wrote {out_path} ({source}, {out_path.stat().st_size:,} bytes)")

    _write_readme(output_dir, readme_rows)
    return written


def main() -> int:
    print(f"Output directory: {OUTPUT_DIR}")
    paths = download_all()
    md_files = sorted(p for p in OUTPUT_DIR.glob("*.md") if p.name != "README.md")
    print(f"\nGenerated {len(md_files)} full-text markdown file(s):")
    for path in md_files:
        print(f"  - {path.name}: {path.stat().st_size:,} bytes")
    if len(md_files) < len(LAWS):
        print(f"ERROR: expected at least {len(LAWS)} .md files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
