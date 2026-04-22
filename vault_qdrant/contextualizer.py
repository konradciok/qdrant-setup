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

        The document block is tagged cache_control=ephemeral so repeated calls
        for different chunks of the same document reuse the cached tokens.

        Args:
            document: Full document text
            chunk: Chunk of text from the document

        Returns:
            Contextual summary + chunk (separated by two newlines)

        Raises:
            anthropic.APIError: If API call fails
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is a document:\n<document>\n{document}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "\n\nSituate this chunk in the document in 1-2 sentences:\n"
                                f"<chunk>\n{chunk}\n</chunk>"
                            ),
                        },
                    ],
                }
            ],
        )

        context = response.content[0].text
        return f"{context}\n\n{chunk}"
