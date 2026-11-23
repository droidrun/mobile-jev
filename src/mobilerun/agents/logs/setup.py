"""Configure agent library log filters."""

import logging

from mobilerun.agents.logs.error_descriptions import HtmlErrorPageFilter

for name in ("llama_index.llms.openai.utils", "llama_index.llms.google_genai.utils"):
    logging.getLogger(name).addFilter(HtmlErrorPageFilter())
