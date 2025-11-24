from langchain_core.messages import BaseMessage


class Context:
    def __init__(self):
        self._messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> "Context":
        self._messages.append(message)
        return self

    def get_messages(self, last_n: int = None) -> list[BaseMessage]:
        if last_n:
            return self._messages[-last_n:]
        return self._messages

    def __str__(self):
        return str(self._messages)
