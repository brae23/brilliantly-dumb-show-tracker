import math
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken
from openai import OpenAI

class HybridSemanticChunker:
    """
    A class for performing hybrid semantic chunking on diarized video transcripts

    - Semantic chunking based on speaker diarization and transcript content, using cosine similarity
    - Includes fallback max_tokens guardrail to ensure chunks do not exceed a certain number of tokens (hence the hybrid approach).
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        window_size: int = 1,
        breakpoint_percentile: float = 80.0,
        max_tokens: int = 400,
        tokenizer_name: str = "cl100k_base",
    ):
        self.client = openai_client or OpenAI()
        self.embedding_model = embedding_model
        self.window_size = window_size
        self.breakpoint_percentile = breakpoint_percentile
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _strip_sentence_text(self, sentence: Dict[str, Any]) -> str:
        return sentence["text"].strip()

    def _join_sentences(self, sentences: List[Dict[str, Any]]) -> str:
        return " ".join(self._strip_sentence_text(s) for s in sentences)

    def _cosine_distance(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(1.0 - similarity)

    def _get_window_embeddings(self, sentences: List[Dict[str, Any]]) -> List[List[float]]:
        print(f"Fetching window embeddings for {len(sentences)} sentences with window size {self.window_size}")
        window_texts = []
        n = len(sentences)

        for i in range(n):
            start_idx = max(0, i - self.window_size)
            end_idx = min(n, i + self.window_size + 1)
            window_str = self._join_sentences(sentences[start_idx:end_idx])
            window_texts.append(window_str)

        print(f"Created {len(window_texts)} window texts for embedding request")
        print(f"Requesting embeddings for {len(window_texts)} window texts from OpenAI")
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=window_texts
        )
        print(f"Received embeddings for {len(response.data)} window texts")
        return [data.embedding for data in response.data]

    def _build_chunk_payload(
        self,
        sentences: List[Dict[str, Any]],
        episode_id: str,
        split_reason: str,
        youtube_base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        text_content = self._join_sentences(sentences)
        
        start_time = sentences[0]["start"]
        end_time = sentences[-1]["end"]
        speaker_name = sentences[0].get("speaker", "UNKNOWN")

        start_seconds = math.floor(start_time)
        youtube_timestamp_url = (
            f"{youtube_base_url}&t={start_seconds}s" if youtube_base_url else None
        )

        return {
            "episode_id": episode_id,
            "text_content": text_content,
            "metadata": {
                "speaker_name": speaker_name,
                "start_time": start_time,
                "end_time": end_time,
                "youtube_timestamp_url": youtube_timestamp_url,
                "split_reason": split_reason,
            }
        }

    def chunk_transcript(
        self,
        sentences: List[Dict[str, Any]],
        episode_id: str,
        youtube_base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main execution pipeline for hybrid semantic chunking (semantic first with hard token cap guardrail).
        
        Args:
            sentences: List of WhisperX utterance segments with keys:
                       'text', 'start', 'end', and optional 'speaker'.
            episode_id: ID of the episode being ingested.
            youtube_base_url: Base YouTube URL for timestamp generation.
        """
        if not sentences:
            return []

        if len(sentences) == 1:
            return [self._build_chunk_payload(sentences, episode_id, "semantic", youtube_base_url)]

        # 1. Generate sliding window embeddings - we need these to compute semantic distances between adjacent segments
        embeddings = self._get_window_embeddings(sentences)

        # 2. Compute adjacent cosine distances - this helps us identify where semantic shifts occur between consecutive segments
        distances = []
        for a, b in zip(embeddings, embeddings[1:]):
            dist = self._cosine_distance(a, b)
            distances.append(dist)

        # 3. Calculate percentile breakpoint threshold - this determines the distance at which we consider a semantic shift significant enough to create a new chunk
        threshold = float(np.percentile(distances, self.breakpoint_percentile))

        # 4. Form chunks based on semantic breakpoints and max_tokens guardrail
        chunks = []
        current_sentences: List[Dict[str, Any]] = []

        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_text = self._join_sentences(current_sentences)
            current_tokens = self._count_tokens(current_text)

            if current_tokens > self.max_tokens:
                if len(current_sentences) > 1:
                    # Spill last sentence over to next chunk
                    overflow_sentence = current_sentences.pop()
                    chunks.append(
                        self._build_chunk_payload(
                            current_sentences, episode_id, "max_tokens", youtube_base_url
                        )
                    )
                    current_sentences = [overflow_sentence]
                else:
                    # Single sentence exceeds max tokens, force chunk creation
                    chunks.append(
                        self._build_chunk_payload(
                            current_sentences, episode_id, "max_tokens", youtube_base_url
                        )
                    )
                    current_sentences = []
                continue

            # Check Semantic Breakpoint (distance after sentence i exceeds threshold)
            if i < len(distances) and distances[i] >= threshold:
                chunks.append(
                    self._build_chunk_payload(
                        current_sentences, episode_id, "semantic", youtube_base_url
                    )
                )
                current_sentences = []

        # Catch remaining sentences
        if current_sentences:
            chunks.append(
                self._build_chunk_payload(
                    current_sentences, episode_id, "semantic", youtube_base_url
                )
            )

        return chunks