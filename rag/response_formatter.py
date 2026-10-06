"""

Response Formatter: Structures and styles RAG responses for clean, professional rendering in Streamlit.

Conforms strictly to existing card styles without modifying the visual design language.

Ensures ZERO raw HTML source code is ever displayed to the user.

"""

from typing import List, Dict, Any, Union, Optional
import html
import json
import re
from .documents import Document

def format_point_by_point_markdown(text: str) -> str:
    """
    Formats a veterinary RAG answer ensuring strict point-by-point presentation in Streamlit:
    1. Every numbered point starts on a new line with a blank line between numbered points.
    2. Every bullet point appears on its own separate line with a blank line between points.
    3. Inline numbered points and inline bullets joined on a single line are cleanly split.
    4. Escaped newlines (\\n) are converted to actual newlines.
    5. Raw JSON structures and accidental HTML tags are sanitized and never exposed.
    6. Headings and horizontal rules are preserved as separate blocks.
    7. English and Telugu answers maintain the identical point-by-point structure.
    8. Preserves the exact medical content and original order without modification.
    """
    if not text:
        return ""

    s = str(text)

    # 1. Convert escaped newlines
    s = s.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")

    # 2. Check if text is raw JSON and extract clean content
    trimmed = s.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                blocks = []
                if "english" in parsed and isinstance(parsed["english"], list):
                    for idx, sec in enumerate(parsed["english"], start=1):
                        title = sec.get("title") or sec.get("heading") or f"Section {idx}"
                        content = sec.get("content") or ""
                        blocks.append(f"### {idx}. {title}\n\n{content}")
                    s = "\n\n---\n\n".join(blocks)
                elif "telugu" in parsed and isinstance(parsed["telugu"], list):
                    for idx, sec in enumerate(parsed["telugu"], start=1):
                        title = sec.get("title") or sec.get("heading") or f"విభాగం {idx}"
                        content = sec.get("content") or ""
                        blocks.append(f"### {idx}. {title}\n\n{content}")
                    s = "\n\n---\n\n".join(blocks)
                elif "content" in parsed:
                    s = str(parsed["content"])
                elif "answer" in parsed:
                    s = str(parsed["answer"])
        except Exception:
            pass

    # 3. Sanitize HTML tags while preserving line breaks
    s = re.sub(r"<br\s*/?>", "\n\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<li[^>]*>(.*?)</li>", r"\n• \1\n", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</?(?:ul|ol|div|span|p|section|article)[^>]*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<a\s+[^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", r"[\2](\1)", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)

    # 4. Split inline bullets that were joined on a single line:
    # e.g., "• Keep sheep clean. • Give clean water. • Monitor temperature."
    s = re.sub(r"(\S)\s*[•●*]\s+", r"\1\n\n• ", s)
    s = re.sub(r"^[•●*]\s+", r"• ", s, flags=re.MULTILINE)

    # 5. Split inline numbered points that were joined into one paragraph:
    # e.g., "1. Keep the sheep clean and dry. 2. Provide clean water. 3. Monitor symptoms."
    s = re.sub(r"([.!?:\'\"\u0C00-\u0C7F])\s+(\d+|[\u0C66-\u0C6F]+)[\.\)]\s+", r"\1\n\n\2. ", s)
    s = re.sub(r"([^#\s])\s+(\d+|[\u0C66-\u0C6F]+)[\.\)]\s+([A-Z\u0C00-\u0C7F*])", r"\1\n\n\2. \3", s)

    # 6. Ensure headings have blank lines before and after them
    s = re.sub(r"(?<!\n)\n(#{1,4}\s+)", r"\n\n\1", s)
    s = re.sub(r"(#{1,4}\s+[^\n]+)\n(?!\n)", r"\1\n\n", s)

    # 7. Ensure horizontal rules have clean blank lines
    s = re.sub(r"(?<!\n)\n(---+)\n", r"\n\n\1\n\n", s)

    # 8. Process line by line to guarantee blank line between numbered points and bullets
    lines = [line.strip() for line in s.splitlines()]
    processed_lines = []

    for i, line in enumerate(lines):
        if not line:
            if processed_lines and processed_lines[-1] != "":
                processed_lines.append("")
            continue

        is_heading = bool(re.match(r"^#{1,4}\s+", line))
        is_hr = line == "---"
        is_bullet = bool(re.match(r"^[•●*]\s+", line))
        is_numbered = bool(re.match(r"^(\d+|[\u0C66-\u0C6F]+)[\.\)]\s+", line))

        # Check if we should insert a blank line before this item
        if processed_lines and processed_lines[-1] != "":
            prev_line = processed_lines[-1]
            prev_is_heading = bool(re.match(r"^#{1,4}\s+", prev_line))
            prev_is_bullet = bool(re.match(r"^[•●*]\s+", prev_line))
            prev_is_numbered = bool(re.match(r"^(\d+|[\u0C66-\u0C6F]+)[\.\)]\s+", prev_line))

            if is_heading or is_hr or prev_is_heading or (prev_line == "---") or is_numbered or is_bullet:
                processed_lines.append("")

        processed_lines.append(line)

    result = "\n".join(processed_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()

def format_retrieved_sources_markdown(documents: List[Document], language: str = "English + తెలుగు") -> str:

    """

    Renders clickable, verified veterinary knowledge source tags in native Markdown.

    Native markdown avoids all Streamlit CommonMark HTML indentation escaping issues.

    """

    if not documents:

        return ""

    seen_sources = set()

    links = []

    for doc in documents:

        src_name = doc.metadata.get("source", "ICAR-IVRI Reference")

        src_url = doc.metadata.get("source_url", "https://ivri.nic.in")

        doc_name = doc.metadata.get("document_name", "Clinical Guideline")

        sec_name = doc.metadata.get("section", "")

        key = (src_name, sec_name)

        if key in seen_sources:

            continue

        seen_sources.add(key)

        label = f"{src_name} ({sec_name})" if sec_name else f"{src_name} — {doc_name}"

        links.append(f"• [{label}]({src_url})")

    if not links:

        return ""

    title = "Verified Veterinary Sources" if language == "English" else ("ధృవీకరించబడిన పశువైద్య వనరులు" if language == "తెలుగు" else "Verified Veterinary Sources / ధృవీకరించబడిన పశువైద్య వనరులు")

    return f"**📚 {title}:**\n" + "\n".join(links[:4])

def sanitize_sources_to_markdown(sources_input: Union[List[Document], str, Any]) -> str:

    """

    Normalizes any sources input (list of Document objects, raw HTML string, or text)

    into clean, safe, clickable Markdown bullet points.

    Never exposes raw HTML tags like <div>, <ul>, <li>, <a href="..."> to the user.

    """

    if not sources_input:

        return ""

    if isinstance(sources_input, list):

        return format_retrieved_sources_markdown(sources_input)

    if isinstance(sources_input, str):

        # Check if the string contains HTML <a> tags

        html_links = re.findall(r"<a\s+[^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", sources_input, re.DOTALL | re.IGNORECASE)

        if html_links:

            md_lines = ["**📚 Verified Veterinary Sources:**"]

            for url, label in html_links:

                clean_label = re.sub(r"<[^>]+>", "", label).strip()

                if clean_label and url:

                    md_lines.append(f"• [{clean_label}]({url})")

            return "\n".join(md_lines)

        clean_text = re.sub(r"<[^>]+>", " ", sources_input).strip()

        if not clean_text:

            return ""

        if clean_text.startswith("**📚"):

            return clean_text

        return f"**📚 Verified Veterinary Sources:**\n{clean_text}"

    return ""

def format_retrieved_sources_html(documents: List[Document], language: str = "English + తెలుగు") -> str:

    """

    Renders clickable, verified veterinary knowledge source tags in HTML.

    STRICT REQUIREMENT: Every line starts with zero leading spaces to prevent

    CommonMark from treating it as an indented code block (<pre><code>).

    """

    if not documents:

        return ""

    seen_sources = set()

    links = []

    for doc in documents:

        src_name = doc.metadata.get("source", "ICAR-IVRI Reference")

        src_url = doc.metadata.get("source_url", "https://ivri.nic.in")

        doc_name = doc.metadata.get("document_name", "Clinical Guideline")

        sec_name = doc.metadata.get("section", "")

        key = (src_name, sec_name)

        if key in seen_sources:

            continue

        seen_sources.add(key)

        label = f"{src_name} ({sec_name})" if sec_name else f"{src_name} — {doc_name}"

        escaped_label = html.escape(label)

        links.append(f'<li><a href="{src_url}" target="_blank" style="color:#0284C7; text-decoration:none;">{escaped_label}</a></li>')

    if not links:

        return ""

    title = "Verified Veterinary Sources Grounding this Answer:" if language == "English" else "ధృవీకరించబడిన పశువైద్య వనరులు:"

    # Zero leading indentation on every line

    lines = [

        '<div style="margin-top:10px; padding:10px 12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px;">',

        f'<div style="font-size:0.8rem; font-weight:700; color:#0A3273; margin-bottom:4px;">📚 {title}</div>',

        '<ul style="margin:0; padding-left:18px; font-size:0.75rem; color:#475569;">',

        "".join(links[:4]),

        '</ul>',

        '</div>'

    ]

    return "\n".join(lines)

def render_chat_message_html(role: str, content: str, sources_html: str = "") -> str:

    """

    Renders user and assistant messages matching existing card and chat bubble styling.

    STRICT REQUIREMENT: Guaranteed zero leading spaces on all lines so Markdown

    never converts the markup into a code block.

    """

    if role == "user":

        escaped = html.escape(content.strip())

        return f'<div class="chat-bubble-user">{escaped}</div>'

    # Assistant message

    # Normalize sources: if sources_html is present, clean it

    clean_sources_md = sanitize_sources_to_markdown(sources_html)

    # Check if content has both English and Telugu sections

    en_part = ""

    te_part = ""

    if "\n\n---\n\n" in content:

        parts = content.split("\n\n---\n\n", 1)

        en_part = parts[0].strip()

        te_part = parts[1].strip()

    elif "---TELUGU_TRANSLATION---" in content:

        parts = content.split("---TELUGU_TRANSLATION---", 1)

        en_part = parts[0].strip()

        te_part = parts[1].strip()

    elif "## Telugu Guidance" in content:

        parts = content.split("## Telugu Guidance", 1)

        en_part = parts[0].strip()

        te_part = "## Telugu Guidance\n" + parts[1].strip()

    else:

        en_part = content.strip()

    # Convert sources to clean HTML if present

    sources_block = ""

    if sources_html:

        # Check if already clean HTML

        if "<div" in sources_html:

            # Strip all line indentation

            sources_block = "\n".join(line.strip() for line in sources_html.strip().splitlines() if line.strip())

        else:

            sources_block = f'<div style="font-size:0.75rem; color:#475569; margin-top:8px;">{html.escape(sources_html)}</div>'

    if en_part and te_part:

        # Side-by-side bilingual display

        lines = [

            '<div class="chat-bubble-assistant">',

            '<div style="font-weight:700; color:#0A3273; font-size:0.85rem; margin-bottom:8px; display:flex; align-items:center; gap:6px;">',

            '<span>🩺</span> Grounded Veterinary Assistant',

            '</div>',

            '<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:6px;">',

            '<!-- English Side -->',

            '<div style="background:#FFFFFF; border:1px solid #D8E6F5; border-radius:6px; padding:12px; font-size:0.83rem; color:#1E293B; line-height:1.55;">',

            '<div style="font-weight:700; color:#0056D2; font-size:0.8rem; margin-bottom:6px; border-bottom:1.5px solid #0056D2; padding-bottom:3px; display:flex; align-items:center; gap:5px;">',

            '<span>🇬🇧</span> English',

            '</div>',

            f'<div style="white-space:pre-wrap;">{html.escape(en_part)}</div>',

            '</div>',

            '<!-- Telugu Side -->',

            '<div style="background:#FFFFFF; border:1px solid #D8E6F5; border-radius:6px; padding:12px; font-size:0.83rem; color:#1E293B; line-height:1.55;">',

            '<div style="font-weight:700; color:#0056D2; font-size:0.8rem; margin-bottom:6px; border-bottom:1.5px solid #0056D2; padding-bottom:3px; display:flex; align-items:center; gap:5px;">',

            '<span>🇮🇳</span> తెలుగు (Telugu)',

            '</div>',

            f'<div style="white-space:pre-wrap;">{html.escape(te_part)}</div>',

            '</div>',

            '</div>',

            sources_block,

            '</div>'

        ]

        return "\n".join([line for line in lines if line])

    else:

        # Monolingual / general display

        disp_text = en_part or content.strip()

        lines = [

            '<div class="chat-bubble-assistant">',

            '<div style="font-weight:700; color:#0A3273; font-size:0.82rem; margin-bottom:4px; display:flex; align-items:center; gap:5px;">',

            '<span>🩺</span> Veterinary Knowledge Assistant',

            '</div>',

            f'<div style="line-height:1.55; white-space:pre-wrap;">{html.escape(disp_text)}</div>',

            sources_block,

            '</div>'

        ]

        return "\n".join([line for line in lines if line])