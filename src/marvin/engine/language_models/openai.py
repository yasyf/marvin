import inspect
from logging import Logger
from typing import Callable, Union

import openai
import openai.openai_object

import marvin
import marvin.utilities.types
from marvin.utilities.async_utils import create_task
from marvin.utilities.logging import get_logger
from marvin.utilities.messages import Message, Role

from .base import ChatLLM, OpenAIFunction, StreamHandler

CONTEXT_SIZES = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-16k-0613": 16384,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
}


def openai_role_map(marvin_role: Role) -> str:
    if marvin_role == Role.FUNCTION_RESPONSE:
        return "function"
    elif marvin_role == Role.FUNCTION_REQUEST:
        return "assistant"
    else:
        return marvin_role.value.lower()


class OpenAIStreamHandler(StreamHandler):
    async def handle_streaming_response(
        self,
        api_response: openai.openai_object.OpenAIObject,
    ) -> Message:
        """
        Accumulate chunk deltas into a full response. Returns the full message.
        Passes partial messages to the callback, if provided.
        """
        response = {"role": None, "content": "", "data": {}, "llm_response": None}

        async for r in api_response:
            response["llm_response"] = r.to_dict_recursive()

            delta = r.choices[0].delta

            if "role" in delta:
                response["role"] = delta.role

            if fn_call := delta.get("function_call"):
                if "function_call" not in response["data"]:
                    response["data"]["function_call"] = {"name": None, "arguments": ""}
                if "name" in fn_call:
                    response["data"]["function_call"]["name"] = fn_call.name
                if "arguments" in fn_call:
                    response["data"]["function_call"]["arguments"] += (
                        fn_call.arguments or ""
                    )

            if "content" in delta:
                response["content"] += delta.content or ""

            if self.callback:
                callback_result = self.callback(Message(**response))
                if inspect.isawaitable(callback_result):
                    create_task(callback_result)

        return Message(**response)


class OpenAIChatLLM(ChatLLM):
    @property
    def context_size(self) -> int:
        return CONTEXT_SIZES.get(self.model, 4096)

    def format_messages(
        self, messages: list[Message]
    ) -> Union[str, dict, list[Union[str, dict]]]:
        """Format Marvin message objects into a prompt compatible with the LLM model"""
        formatted_messages = []
        for m in messages:
            role = openai_role_map(m.role)
            fmt = {"role": role, "content": m.content}
            if m.name:
                fmt["name"] = m.name
            formatted_messages.append(fmt)
        return formatted_messages

    async def run(
        self,
        messages: list[Message],
        *,
        functions: list[OpenAIFunction] = None,
        function_call: Union[str, dict[str, str]] = None,
        logger: Logger = None,
        stream_handler: Callable[[Message], None] = False,
        **kwargs,
    ) -> Message:
        """Calls an OpenAI LLM with a list of messages and returns the response."""

        # ----------------------------------
        # Validate arguments
        # ----------------------------------

        if functions is None:
            functions = []
        if function_call is None:
            function_call = "auto"
        elif function_call not in (
            ["auto", "none"] + [{"name": f.name} for f in functions]
        ):
            raise ValueError(f"Invalid function_call value: {function_call}")
        if logger is None:
            logger = get_logger(self.name)

        # ----------------------------------
        # Form OpenAI-specific arguments
        # ----------------------------------

        prompt = self.format_messages(messages)
        llm_functions = [f.dict(exclude={"fn"}, exclude_none=True) for f in functions]

        # only add to kwargs if supplied, because empty parameters are not
        # allowed by OpenAI
        if functions:
            kwargs["functions"] = llm_functions
            kwargs["function_call"] = function_call

        # ----------------------------------
        # Call OpenAI LLM
        # ----------------------------------

        if not marvin.settings.openai.api_key:
            raise ValueError(
                "OpenAI API key not set. Please set it or use the MARVIN_OPENAI_API_KEY"
                " environment variable."
            )
        response = await openai.ChatCompletion.acreate(
            api_key=marvin.settings.openai.api_key.get_secret_value(),
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True if stream_handler else False,
            **kwargs,
        )

        if stream_handler:
            handler = OpenAIStreamHandler(callback=stream_handler)
            msg = await handler.handle_streaming_response(response)
            return msg

        else:
            llm_response = response.to_dict_recursive()
            msg = llm_response["choices"][0]["message"].copy()
            role = msg.pop("role").upper()
            if role == "ASSISTANT" and isinstance(msg.get("function_call"), dict):
                role = Role.FUNCTION_REQUEST
            msg = Message(
                role=role,
                content=msg.pop("content"),
                data=msg,
                llm_response=llm_response,
            )
            return msg
