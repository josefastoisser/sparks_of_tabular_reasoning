import os
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI

logger = logging.getLogger("evaluation")


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        logger.info(f"Loading remote chat model: {model}")

        self.chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.model, self.tok = None, None

    def __call__(self, chat_msgs: List[Dict]) -> str:
        return self.chat.invoke(chat_msgs).content


if __name__ == "__main__":
    api_key = os.getenv("CAND_API_KEY", "")
    base_url = os.getenv("CAND_BASE_URL", "")
    llm = LLM(model="Qwen/Qwen2.5-14B-Instruct", api_key=api_key, base_url=base_url)

    response = llm(
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "How are you?"},
        ]
    )
    print(response)





