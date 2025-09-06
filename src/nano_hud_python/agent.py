# pyright: reportUnknownArgumentType=false
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
                logger.error(f"Unknown content block type: {block}")
                raise NotImplementedError(f"Unknown content block type: {block}")
    return messages


class Agent(BaseModel):
    """Litellm agent that wraps a Docker environment."""

    model: str
    input: list[ResponseInputItemParam]

    def __init__(self, model: str, *, content: list[ContentBlock] | None = None):
        logger.info(f"Initializing Agent with model: {model}")
        logger.debug(f"Content blocks provided: {len(content) if content else 0}")
        super().__init__(
            model=model,
            input=transform_content(content) if content else [],
        )

    async def run(self, environment: Environment):
        """Run the agent."""
        logger.info(f"Starting agent run with environment: {environment.container_name}")
        mcp_tools = await environment.list_tools()
        logger.info(f"Retrieved {len(mcp_tools)} MCP tools from environment")
        logger.debug(f"MCP tools details: {[tool.name for tool in mcp_tools]}")
        logger.debug("Converting MCP tools to OpenAI format")
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
        logger.debug(f"Converted {len(oai_tools)} tools to OpenAI format")
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Starting agent iteration {iteration}")
            logger.debug(f"Current input length: {len(self.input)} items")
            
            response = await litellm.aresponses(
                model=self.model,
                tools=oai_tools,
                input=self.input,
            )
            assert isinstance(response, ResponsesAPIResponse)
            logger.debug(f"Received response with {len(response.output)} output items")

            # store the tool call results till all tool calls are completed
            tool_call_results: list[FunctionCallOutput] = []
            logger.debug(f"Processing {len(response.output)} response items")

            for idx, item in enumerate(response.output):
                logger.debug(f"Processing item {idx+1}/{len(response.output)}: {type(item).__name__}")
                match item:
                    case ResponseOutputMessage() | GenericResponseOutputItem():
                        logger.debug(f"Model output message: {item.model_dump_json()}")
                        self.input.append(ResponseOutputMessageParam(**item.model_dump()))
                    case ResponseReasoningItem():
                        logger.debug(f"Model reasoning item: {item.model_dump_json()}")
                        self.input.append(ResponseReasoningItemParam(**item.model_dump()))
                    case ResponseFunctionToolCall() | OutputFunctionToolCall():
                        logger.info(f"Processing tool call: {item.name}")
                        self.input.append(ResponseFunctionToolCallParam(**item.model_dump()))

                        assert item.name is not None
                        assert item.arguments is not None
                        assert item.call_id is not None
                        
                        logger.debug(f"Tool call {item.call_id}: {item.name} with arguments: {item.arguments}")
                        result = await environment.call_tool(item.name, json.loads(item.arguments))
                        logger.debug(f"Tool call {item.call_id} completed successfully")
                        tool_call_results.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=json.dumps(result.structuredContent),
                            )
                        )
                        logger.debug(f"Stored result for tool call {item.call_id}")
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
                        logger.error(f"Unsupported tool call type: {type(item).__name__}")
                        raise NotImplementedError(f"Tool call {item.model_dump_json()} is not supported yet")
                    case dict():
                        logger.error("Unknown dictionary item type encountered")
                        raise NotImplementedError(f"Unknown item type: {json.dumps(item)}")

            # add the tool call results to the input now that the outputs are all added
            if tool_call_results:
                logger.info(f"Adding {len(tool_call_results)} tool call results to input")
                self.input.extend(tool_call_results)
            else:
                logger.info(f"No tool calls in iteration {iteration}, agent run complete")

            if len(tool_call_results) == 0:
                break
        
        logger.info(f"Agent run completed after {iteration} iterations")
