import asyncio
import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Literal, override
from uuid import uuid4

from mcp import ClientSession, ListToolsResult, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Environment(BaseModel, AbstractAsyncContextManager["Environment"]):
    """RL environment that wraps a Docker image."""

    # Docker arguments
    image: str
    command: list[str] | None
    publish_all: bool = False
    ports: list[int] | None
    volumes: list[str] | None
    env: list[tuple[str, str]] | None
    runtime: str | None
    memory: str | None
    cpu: str | None
    container_name: str
    # delete mode
    delete_mode: Literal["delete", "stop", "leave"]
    # setup/evaluate tool settings
    setup_tool: str | None
    evaluate_tool: str | None
    # tool expose settings
    allowed_tools: list[str] | None



    # Runtime attributes (not part of initialization)
    _exit_stack: AsyncExitStack | None = None
    _session: ClientSession | None = None

    def __init__(
        self,
        image: str,
        *,
        # docker settings
        command: list[str] | None = None,
        publish_all: bool = False,
        ports: list[int] | None = None,
        volumes: list[str] | None = None,
        env: list[tuple[str, str]] | None = None,
        runtime: str | None = None,
        memory: str | None = None,
        cpu: str | None = None,
        container_name: str | None = None,
        # delete mode
        delete_mode: Literal["delete", "stop", "leave"] = "delete",
        # auto setup/evaluate tool settings
        setup_tool: str | None = None,
        evaluate_tool: str | None = None,
        # tool expose settings
        allowed_tools: list[str] | None = None,
    ):
        """Initialize the Docker environment with specified parameters.

        Args:
            image: Docker image to run
            command: Command to run in the container (default: None)
            publish_all: Publish all exposed ports (default: False)
            ports: List of ports to expose (default: None)
            volumes: List of volume mounts in format "host:container:options" or "host:container" (default: None)
            env: Environment variables as list of tuples (default: None)
            runtime: Docker runtime to use (default: None)
            memory: Memory limit (e.g., "8g") (default: None)
            cpu: CPU limit (e.g., "4") (default: None)
            container_name: Name of the container (default: None)
            delete_mode: Delete mode (default: "delete")
            setup_tool: Name of the setup tool (default: None)
            evaluate_tool: Name of the evaluate tool (default: None)
            allowed_tools: List of tools to include (default: None)
        """
        super().__init__(
            image=image,
            command=command,
            publish_all=publish_all,
            ports=ports,
            volumes=volumes,
            env=env,
            runtime=runtime,
            memory=memory,
            cpu=cpu,
            container_name=container_name or f"nano-hud-python-{uuid4()}",
            delete_mode=delete_mode,
            setup_tool=setup_tool,
            evaluate_tool=evaluate_tool,
            allowed_tools=allowed_tools,
        )

    def _docker_run_args(self) -> list[str]:
        # Build docker run command
        args = ["run", "-i", "--name", self.container_name]

        # Add publish-all flag if specified
        if self.publish_all:
            args.append("--publish-all")

        # Add specific ports if provided
        if self.ports:
            for port in self.ports:
                args.extend(["-p", str(port)])

        # Add environment variables
        if self.env:
            for key, value in self.env:
                args.extend(["-e", f"{key}={value}"])

        # Add resource limits
        if self.cpu:
            args.append(f"--cpus={self.cpu}")

        if self.memory:
            args.append(f"--memory={self.memory}")

        # Add runtime if specified
        if self.runtime:
            args.append(f"--runtime={self.runtime}")

        # Add volume mounts
        if self.volumes:
            for volume in self.volumes:
                args.extend(["-v", volume])

        # Add the image
        args.append(self.image)

        # Add command if specified
        if self.command:
            args.extend(self.command)

        return args

    async def initialize(self):
        """Initialize and start the Docker container."""
        self._exit_stack = AsyncExitStack()
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(command="docker", args=self._docker_run_args(), env=None)),
        )
        stdio, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(stdio, write))

        # Initialize
        _ = await self._session.initialize()

        logger.info(f"Docker container {self.container_name} started successfully")

        # now attempt to connect to mcp server running inside
        self._exit_stack = AsyncExitStack()

    async def cleanup(self):
        """Clean up resources and handle Docker container based on delete_mode"""
        assert self._exit_stack is not None
        await self._exit_stack.aclose()
        if self.container_name:
            if self.delete_mode == "delete":
                # Remove the container completely
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "rm",
                        "-f",
                        self.container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _ = await proc.communicate()
                except Exception as e:
                    logger.error(f"Failed to remove container {self.container_name}: {e}")
            elif self.delete_mode == "stop":
                # Stop the container but don't remove it
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "stop",
                        self.container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _ = await proc.communicate()
                except Exception as e:
                    logger.error(f"Failed to stop container {self.container_name}: {e}")
            elif self.delete_mode == "leave":
                # Do nothing - leave the container running
                pass

    async def list_all_tools(self) -> list[Tool]:
        assert self._session is not None
        # list all tools
        all_tools: list[Tool] = []
        cursor = None
        while True:
            result: ListToolsResult = await self._session.list_tools(cursor)
            all_tools.extend(result.tools)
            if result.nextCursor is None:
                break
            cursor = result.nextCursor

        return all_tools

    async def list_tools(self) -> list[Tool]:
        """List tools available from the Docker container, filtered by allowed_tools, and excluding setup_tool and evaluate_tool"""
        all_tools = await self.list_all_tools()
        filtered_tools: list[Tool] = []
        for tool in all_tools:
            # exclude setup and evaluate tools
            if tool.name in [self.setup_tool, self.evaluate_tool]:
                continue
            # exclude tools not in allowed_tools (if specified)
            if self.allowed_tools is not None and tool.name not in self.allowed_tools:
                continue
            filtered_tools.append(tool)

        return filtered_tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        """Call a tool available from the Docker container."""
        assert self._session is not None
        return await self._session.call_tool(tool_name, tool_args)

    async def setup(self) -> CallToolResult:
        """Setup the Docker container."""
        assert self.setup_tool is not None
        return await self.call_tool(self.setup_tool, {})

    async def evaluate(self) -> CallToolResult:
        """Evaluate the Docker container."""
        assert self.evaluate_tool is not None
        return await self.call_tool(self.evaluate_tool, {})

    @override
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Async context manager exit."""
        await self.cleanup()
