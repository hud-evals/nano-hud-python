import json
import logging

import litellm
from litellm import GenericResponseOutputItem, OutputFunctionToolCall, ResponsesAPIResponse
from mcp.types import ContentBlock, ImageContent, TextContent
from openai.types.responses import (
    FunctionToolParam,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import BaseModel

from nano_hud_python.json_schema import init_strict_json_schema

from .environment import Environment

logger = logging.getLogger(__name__)


def transform_content(content: list[ContentBlock]) -> list[Message]:
    messages: list[Message] = []
    for block in content:
        match block:
            case TextContent():
                messages.append(
                    Message(role="user", content=[ResponseInputTextParam(type="input_text", text=block.text)])
                )
            case ImageContent():
                messages.append(
                    Message(
                        role="user",
                        content=[ResponseInputImageParam(type="input_image", detail="high", image_url=block.data)],
                    )
                )
            case _:
                raise NotImplementedError(f"Unknown content block type: {block}")
    return messages


class Agent(BaseModel):
    """Litellm agent that wraps a Docker environment."""

    model: str
    input: list[ResponseInputItemParam]

    def __init__(self, model: str, *, content: list[ContentBlock] | None = None):
        super().__init__(
            model=model,
            input=transform_content(content) if content else [],
        )

    async def run(self, environment: Environment):
        """Run the agent."""
        mcp_tools = await environment.list_tools()
        logger.info(f"MCP tools: {mcp_tools}")
        oai_tools = [
            FunctionToolParam(
                name=tool.name,
                description=tool.description,
                parameters=init_strict_json_schema(tool.inputSchema).model_dump(exclude_none=True),
                strict=True,
                type="function",
            )
            for tool in mcp_tools
        ]
        while True:
            response = await litellm.aresponses(
                model=self.model,
                tools=oai_tools,
                input=self.input,
            )
            assert isinstance(response, ResponsesAPIResponse)

            # store the tool call results till all tool calls are completed
            tool_call_results: list[FunctionCallOutput] = []

            for item in response.output:
                match item:
                    case ResponseOutputMessage() | GenericResponseOutputItem():
                        self.input.append(ResponseOutputMessageParam(**item.model_dump()))
                    case ResponseReasoningItem():
                        self.input.append(ResponseReasoningItemParam(**item.model_dump()))
                    case ResponseFunctionToolCall() | OutputFunctionToolCall():
                        self.input.append(ResponseFunctionToolCallParam(**item.model_dump()))

                        assert item.name is not None
                        assert item.arguments is not None
                        assert item.call_id is not None

                        result = await environment.call_tool(item.name, json.loads(item.arguments))
                        tool_call_results.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=json.dumps(result.structuredContent),
                            )
                        )
                    case (
                        ResponseComputerToolCall()
                        | ResponseFileSearchToolCall()
                        | ResponseFunctionWebSearch()
                        | ImageGenerationCall()
                        | ResponseCodeInterpreterToolCall()
                        | LocalShellCall()
                        | McpCall()
                        | McpListTools()
                        | McpApprovalRequest()
                        | ResponseCustomToolCall()
                    ):
                        raise NotImplementedError(f"Tool call {item.model_dump_json()} is not supported yet")
                    case dict():
                        raise NotImplementedError(f"Unknown item type: {json.dumps(item)}")

            # add the tool call results to the input now that the outputs are all added
            self.input.extend(tool_call_results)

            if len(tool_call_results) == 0:
                break
