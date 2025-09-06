import json

import litellm
from litellm import GenericResponseOutputItem, OutputFunctionToolCall, ResponsesAPIResponse
from openai.types.responses import (
    FunctionToolParam,
    ResponseComputerToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
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

            tool_call_results: list[FunctionCallOutput] = []

            for item in response.output:
                match item:
                    # openai types
                    case ResponseOutputMessage() | GenericResponseOutputItem():
                        self.input.append(ResponseOutputMessageParam(**item.model_dump()))
                    case ResponseFunctionToolCall() | OutputFunctionToolCall():
                        self.input.append(ResponseFunctionToolCallParam(**item.model_dump()))

                        # assert the required fields are not None
                        assert item.name is not None
                        assert item.arguments is not None
                        assert item.call_id is not None

                        # call the tool
                        result = await environment.call_tool(item.name, json.loads(item.arguments))
                        tool_call_results.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=json.dumps(result.structuredContent),
                            )
                        )
                    case ResponseComputerToolCall():
                        raise NotImplementedError("Computer tool calls are not supported yet")
                    case _:
                        raise ValueError(f"Unknown item type: {item}")

            # add the tool call results to the input
            self.input.extend(tool_call_results)

            # if there were no tool calls, break
            if len(tool_call_results) == 0:
                break
