"""
Integration tests — Ingestion Pipeline

These tests mock external calls (YouTube, OpenAI) but exercise
the full pipeline flow: clean → chunk → embed → store.

Marked with @pytest.mark.integration.
Run with: pytest tests/integration/ -m integration
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


pytestmark = pytest.mark.integration


class TestCleanToChunkPipeline:
    """Cleaner output feeds correctly into the chunker."""

    def test_cleaned_text_produces_valid_chunks(
        self, sample_raw_transcript, sample_segments
    ):
        from app.services.youtube.transcript_cleaner import TranscriptCleaner
        from app.services.youtube.chunker import TranscriptChunker

        cleaner = TranscriptCleaner()
        chunker = TranscriptChunker(chunk_size=60, overlap=10)

        cleaned = cleaner.clean(sample_raw_transcript)
        assert cleaned.cleaned_text, "Cleaner must produce non-empty text"

        chunks = chunker.chunk(
            cleaned_text=cleaned.cleaned_text,
            segments=cleaned.segments,
            video_id="vid_001",
            video_title="Test Video",
            course_id="course_001",
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.chunk_text.strip()
            assert chunk.token_count > 0
            assert chunk.start_timestamp >= 0
            assert chunk.end_timestamp >= chunk.start_timestamp

    def test_chunk_ids_are_unique_across_pipeline(self, sample_raw_transcript):
        from app.services.youtube.transcript_cleaner import TranscriptCleaner
        from app.services.youtube.chunker import TranscriptChunker

        cleaner = TranscriptCleaner()
        chunker = TranscriptChunker()

        cleaned = cleaner.clean(sample_raw_transcript)
        chunks  = chunker.chunk(
            cleaned_text=cleaned.cleaned_text,
            segments=cleaned.segments,
            video_id="vid_001",
            video_title="Test Video",
            course_id="course_001",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_positions_are_sequential(self, sample_raw_transcript):
        from app.services.youtube.transcript_cleaner import TranscriptCleaner
        from app.services.youtube.chunker import TranscriptChunker

        cleaned = TranscriptCleaner().clean(sample_raw_transcript)
        chunks  = TranscriptChunker().chunk(
            cleaned.cleaned_text, cleaned.segments, "v", "T", "c"
        )
        positions = [c.position for c in chunks]
        assert positions == list(range(len(chunks)))


class TestChunkToEmbedPipeline:
    """Chunks from the chunker are correctly passed to the embedding service."""

    def test_embedding_service_called_with_chunks(
        self, sample_chunks, mock_chroma_collection
    ):
        from app.services.rag.embedding_service import EmbeddingService

        fake_embedding = [0.01] * 1536
        with patch("app.services.rag.embedding_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch("app.services.rag.embedding_service.openai") as mock_oai:

            mock_oai.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=fake_embedding, index=i)
                      for i in range(len(sample_chunks))]
            )

            service = EmbeddingService()
            service.embed_and_store(sample_chunks)

        mock_chroma_collection.upsert.assert_called_once()
        call_kwargs = mock_chroma_collection.upsert.call_args[1]
        assert len(call_kwargs["ids"]) == len(sample_chunks)

    def test_upserted_metadata_contains_required_fields(
        self, sample_chunks, mock_chroma_collection
    ):
        from app.services.rag.embedding_service import EmbeddingService

        fake_embedding = [0.01] * 1536
        with patch("app.services.rag.embedding_service.get_collection",
                   return_value=mock_chroma_collection), \
             patch("app.services.rag.embedding_service.openai") as mock_oai:

            mock_oai.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=fake_embedding, index=i)
                      for i in range(len(sample_chunks))]
            )

            EmbeddingService().embed_and_store(sample_chunks)

        metadatas = mock_chroma_collection.upsert.call_args[1]["metadatas"]
        required_fields = {
            "course_id", "video_id", "video_title",
            "start_timestamp", "end_timestamp", "token_count", "position"
        }
        for meta in metadatas:
            assert required_fields.issubset(meta.keys())


class TestRAGChain:
    """RetrievalService + LLMChain produce a valid ChatResponse."""

    def test_rag_returns_grounded_answer(
        self, sample_retrieved_chunks, mock_llm_response
    ):
        from app.services.rag.llm_chain import LLMChain

        with patch("app.services.rag.llm_chain.openai") as mock_oai:
            mock_oai.chat.completions.create.return_value = mock_llm_response

            chain  = LLMChain()
            result = chain.answer(
                question="What is gradient descent?",
                chunks=sample_retrieved_chunks,
                conversation_history=[],
            )

        assert result.answer
        assert len(result.answer) > 10
        assert isinstance(result.sources, list)
        assert isinstance(result.chunks_used, int)
        assert result.chunks_used == len(sample_retrieved_chunks)

    def test_empty_chunks_returns_not_grounded(self):
        from app.services.rag.llm_chain import LLMChain

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="I don't have information about that in this course.")
        )]

        with patch("app.services.rag.llm_chain.openai") as mock_oai:
            mock_oai.chat.completions.create.return_value = mock_response

            chain  = LLMChain()
            result = chain.answer(
                question="What is quantum entanglement?",
                chunks=[],
                conversation_history=[],
            )

        assert result.chunks_used == 0
        assert result.is_grounded is False

    def test_sources_match_chunk_titles(
        self, sample_retrieved_chunks, mock_llm_response
    ):
        from app.services.rag.llm_chain import LLMChain

        with patch("app.services.rag.llm_chain.openai") as mock_oai:
            mock_oai.chat.completions.create.return_value = mock_llm_response

            result = LLMChain().answer(
                question="gradient descent",
                chunks=sample_retrieved_chunks,
                conversation_history=[],
            )

        source_titles = {s.get("video_title") for s in result.sources}
        assert "Lecture 1 — Gradient Descent" in source_titles


class TestChatbotSession:
    """Chatbot maintains session history across turns."""

    def test_session_history_grows_with_turns(self, sample_retrieved_chunks, mock_llm_response):
        from app.services.rag.chatbot import RAGChatbot

        with patch("app.services.rag.chatbot.RetrievalService") as MockRetrieval, \
             patch("app.services.rag.chatbot.LLMChain") as MockChain:

            MockRetrieval.return_value.retrieve.return_value = sample_retrieved_chunks
            MockChain.return_value.answer.return_value = MagicMock(
                answer="Gradient descent minimizes the loss.",
                sources=[],
                is_grounded=True,
                chunks_used=2,
            )

            bot = RAGChatbot()
            session_id = "sess-001"

            bot.ask("What is gradient descent?", "course-abc", session_id)
            bot.ask("How does the learning rate work?", "course-abc", session_id)

            history = bot._sessions.get(session_id, [])
            assert len(history) == 4    # 2 questions + 2 answers

    def test_clear_session_removes_history(self, sample_retrieved_chunks, mock_llm_response):
        from app.services.rag.chatbot import RAGChatbot

        with patch("app.services.rag.chatbot.RetrievalService") as MockRetrieval, \
             patch("app.services.rag.chatbot.LLMChain") as MockChain:

            MockRetrieval.return_value.retrieve.return_value = sample_retrieved_chunks
            MockChain.return_value.answer.return_value = MagicMock(
                answer="Answer.", sources=[], is_grounded=True, chunks_used=2
            )

            bot = RAGChatbot()
            session_id = "sess-002"
            bot.ask("Question?", "course-abc", session_id)
            assert len(bot._sessions.get(session_id, [])) > 0

            bot.clear_session(session_id)
            assert bot._sessions.get(session_id, []) == []
