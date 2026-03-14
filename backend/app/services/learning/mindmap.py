"""
Mind Map Generator
------------------
Generates a hierarchical concept map for a course or video.

Output structure:
  Course/Video Title
  ├── Main Topic 1
  │   ├── Subtopic 1.1
  │   └── Subtopic 1.2
  ├── Main Topic 2
  │   ├── Subtopic 2.1
  │   └── Subtopic 2.2
  └── Main Topic 3

Strategy:
  1. Fetch all chunk texts for the course/video
  2. Ask LLM to extract the concept hierarchy in JSON
  3. Return structured tree + a text rendering for display
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.database.mongodb import get_chunks_collection
from app.services.learning.base import LearningToolsBase

logger = logging.getLogger(__name__)


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class MindMapNode:
    label: str
    children: list["MindMapNode"] = field(default_factory=list)


@dataclass
class MindMapResult:
    title: str
    root: MindMapNode
    text_render: str        # ASCII tree for simple display
    video_id: Optional[str]
    course_id: str


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at analyzing educational content and
extracting concept hierarchies. Always respond in valid JSON only.
No extra text, no markdown fences."""

_MINDMAP_PROMPT = """Analyze these lecture transcript excerpts and extract
a hierarchical mind map of all concepts taught.

{context}

Respond ONLY with a JSON object in this exact format:
{{
  "title": "Main subject of the lecture",
  "topics": [
    {{
      "name": "Main Topic 1",
      "subtopics": ["Subtopic A", "Subtopic B", "Subtopic C"]
    }},
    {{
      "name": "Main Topic 2",
      "subtopics": ["Subtopic A", "Subtopic B"]
    }}
  ]
}}

Rules:
- 4 to 8 main topics
- 2 to 5 subtopics per main topic
- Use the actual terminology from the transcript
- Topics should follow the logical order they appear in the lecture
"""


# ── Service ───────────────────────────────────────────────────────────────────

class MindMapService(LearningToolsBase):
    """
    Generates a concept mind map from transcript chunks.

    Usage:
        service = MindMapService()
        result = await service.generate(course_id="abc123")
        result = await service.generate(course_id="abc123", video_id="xyz")
    """

    async def generate(
        self,
        course_id: str,
        video_id: Optional[str] = None,
    ) -> MindMapResult:
        """
        Generate a mind map for a course or single video.

        Args:
            course_id:  The course to map.
            video_id:   Optional — restrict to a single video.
        """
        chunks = await self._fetch_chunks(course_id, video_id)
        if not chunks:
            raise ValueError("No transcript data found to generate a mind map.")

        title = chunks[0].get("video_title", "Course") if video_id else "Full Course"

        # Sample chunks evenly to stay within context window
        # (use every Nth chunk if there are too many)
        sampled = self._sample_chunks(chunks, max_chunks=20)
        context = self._build_context(sampled)

        raw = self._call_llm(_SYSTEM_PROMPT, _MINDMAP_PROMPT.format(context=context),
                             temperature=0.3, max_tokens=1500)

        root, map_title = self._parse_mindmap(raw, fallback_title=title)
        text_render = self._render_tree(root)

        logger.info(
            f"Mind map generated: '{map_title}' "
            f"({len(root.children)} main topics)"
        )

        return MindMapResult(
            title=map_title,
            root=root,
            text_render=text_render,
            video_id=video_id,
            course_id=course_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_mindmap(raw: str, fallback_title: str) -> tuple[MindMapNode, str]:
        """Parse the LLM JSON response into a MindMapNode tree."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)

            title = data.get("title", fallback_title)
            root = MindMapNode(label=title)

            for topic in data.get("topics", []):
                node = MindMapNode(label=topic.get("name", ""))
                for sub in topic.get("subtopics", []):
                    node.children.append(MindMapNode(label=sub))
                root.children.append(node)

            return root, title

        except Exception as e:
            logger.error(f"Mind map parse error: {e}\nRaw: {raw[:200]}")
            # Return minimal fallback
            root = MindMapNode(label=fallback_title)
            root.children.append(MindMapNode(label="Content could not be parsed"))
            return root, fallback_title

    @staticmethod
    def _render_tree(node: MindMapNode, prefix: str = "", is_last: bool = True) -> str:
        """Render a MindMapNode tree as an ASCII diagram."""
        connector = "└── " if is_last else "├── "
        lines = [prefix + (connector if prefix else "") + node.label]

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            is_child_last = (i == len(node.children) - 1)
            lines.append(
                MindMapService._render_tree(child, child_prefix, is_child_last)
            )
        return "\n".join(lines)

    @staticmethod
    def _sample_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
        """Sample evenly across chunks to respect context window."""
        if len(chunks) <= max_chunks:
            return chunks
        step = len(chunks) // max_chunks
        return chunks[::step][:max_chunks]

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        return "\n\n".join(
            c.get("chunk_text", "") for c in chunks if c.get("chunk_text")
        )

    @staticmethod
    async def _fetch_chunks(
        course_id: str,
        video_id: Optional[str] = None,
    ) -> list[dict]:
        col = get_chunks_collection()
        query = {"course_id": course_id}
        if video_id:
            query["video_id"] = video_id
        cursor = col.find(query).sort([("video_id", 1), ("position", 1)])
        return [doc async for doc in cursor]
