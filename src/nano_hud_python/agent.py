import json

import litellm
from litellm import GenericResponseOutputItem, OutputFunctionToolCall, ResponsesAPIResponse
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
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import BaseModel

from .environment import Environment


class Agent(BaseModel):
    """Litellm agent that wraps a Docker environment."""

    model: str
    input: list[ResponseInputItemParam]

    def __init__(self, model: str, *, input: list[ResponseInputItemParam] | None = None):
        super().__init__()
        self.model = model
        self.input = input or []

    async def run(self, environment: Environment):
        """Run the agent."""
        mcp_tools = await environment.list_tools()
        oai_tools = [
            FunctionToolParam(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
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
