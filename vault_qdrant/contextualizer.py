"""Contextualizer for vault chunks.

Uses Claude API to add contextual prefixes to chunks based on full document.
"""

from __future__ import annotations

from anthropic import Anthropic


class Contextualizer:
    """Add contextual prefix to chunks using Claude API.

    Calls Claude to generate a 1-2 sentence contextual summary of a chunk
    within the context of its full document.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        """Initialize Contextualizer.

        Args:
            api_key: Anthropic API key
            model: Claude model name (default: haiku for cost optimization)
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def contextualize(self, document: str, chunk: str) -> str:
        """Add contextual prefix to a chunk.

        Calls Claude to generate 1-2 sentences situating the chunk within
        the document. Returns the context prepended to the chunk.

        Args:
            document: Full document text
            chunk: Chunk of text from the document

        Returns:
            Contextual summary + chunk (separated by two newlines)

        Raises:
            anthropic.APIError: If API call fails
        """
        prompt = (
            "Here is a document:\n<document>\n"
            f"{document}\n"
            "</document>\n\n"
            "Situate this chunk in the document in 1-2 sentences:\n<chunk>\n"
            f"{chunk}\n"
            "</chunk>"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        context = response.content[0].text
        return f"{context}\n\n{chunk}"
