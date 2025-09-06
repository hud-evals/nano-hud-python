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
    setup_tool_args: dict[str, Any] | None
    evaluate_tool: str | None
    evaluate_tool_args: dict[str, Any] | None
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
        setup_tool_args: dict[str, Any] | None = None,
        evaluate_tool: str | None = None,
        evaluate_tool_args: dict[str, Any] | None = None,
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
        generated_name = container_name or f"nano-hud-python-{uuid4()}"
        logger.info(f"Initializing Environment with Docker image: {image}")
        logger.debug(f"Container name: {generated_name}")
        logger.debug(f"Delete mode: {delete_mode}")
        logger.debug(f"Ports: {ports}, Volumes: {volumes}, Memory: {memory}, CPU: {cpu}")
        
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
            container_name=generated_name,
            delete_mode=delete_mode,
            setup_tool=setup_tool,
            setup_tool_args=setup_tool_args,
            evaluate_tool=evaluate_tool,
            evaluate_tool_args=evaluate_tool_args,
            allowed_tools=allowed_tools,
        )
        logger.debug("Environment initialization complete")

    def _docker_run_args(self) -> list[str]:
        # Build docker run command
        logger.debug("Building Docker run arguments")
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
        
        logger.debug(f"Docker run arguments: {' '.join(args)}")
        return args

    async def initialize(self):
        """Initialize and start the Docker container."""
        logger.info(f"Starting Docker container: {self.container_name}")
        logger.debug(f"Using Docker image: {self.image}")
        self._exit_stack = AsyncExitStack()
        logger.debug("Creating stdio client for Docker container")
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(command="docker", args=self._docker_run_args(), env=None)),
        )
        logger.debug("Stdio client created successfully")
        logger.debug("Creating MCP client session")
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))

        # Initialize
        logger.info("Initializing MCP session with Docker container")
        result = await self._session.initialize()
        logger.info(f"Docker container {self.container_name} initialized successfully")
        logger.debug(f"Initialization result: {result}")

    async def cleanup(self):
        """Clean up resources and handle Docker container based on delete_mode"""
        logger.info(f"Starting cleanup for container: {self.container_name}")
        assert self._exit_stack is not None
        logger.debug("Closing exit stack")
        await self._exit_stack.aclose()
        if self.container_name:
            logger.debug(f"Cleanup mode: {self.delete_mode}")
            if self.delete_mode == "delete":
                # Remove the container completely
                logger.info(f"Removing container: {self.container_name}")
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
                    logger.info(f"Container {self.container_name} removed successfully")
                except Exception as e:
                    logger.error(f"Failed to remove container {self.container_name}: {e}")
            elif self.delete_mode == "stop":
                # Stop the container but don't remove it
                logger.info(f"Stopping container: {self.container_name}")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "stop",
                        self.container_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _ = await proc.communicate()
                    logger.info(f"Container {self.container_name} stopped successfully")
                except Exception as e:
                    logger.error(f"Failed to stop container {self.container_name}: {e}")
            elif self.delete_mode == "leave":
                # Do nothing - leave the container running
                logger.info(f"Leaving container {self.container_name} running as requested")
                pass

    async def list_all_tools(self) -> list[Tool]:
        assert self._session is not None
        logger.debug("Listing all available tools from container")
        # list all tools
        all_tools: list[Tool] = []
        cursor = None
        page = 0
        while True:
            page += 1
            result: ListToolsResult = await self._session.list_tools(cursor)
            all_tools.extend(result.tools)
            logger.debug(f"Retrieved {len(result.tools)} tools in page {page}")
            if result.nextCursor is None:
                break
            cursor = result.nextCursor
        
        logger.debug(f"Total tools retrieved: {len(all_tools)}")
        return all_tools

    async def list_tools(self) -> list[Tool]:
        """List tools available from the Docker container, filtered by allowed_tools, and excluding setup_tool and evaluate_tool"""
        all_tools = await self.list_all_tools()
        filtered_tools: list[Tool] = []
        for tool in all_tools:
            # exclude setup and evaluate tools
            if tool.name in [self.setup_tool, self.evaluate_tool]:
                logger.debug(f"Excluding setup/evaluate tool: {tool.name}")
                continue
            # exclude tools not in allowed_tools (if specified)
            if self.allowed_tools is not None and tool.name not in self.allowed_tools:
                logger.debug(f"Excluding tool not in allowed list: {tool.name}")
                continue
            filtered_tools.append(tool)
        
        logger.debug(f"Available tools: {[tool.name for tool in filtered_tools]}")
        return filtered_tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        """Call a tool available from the Docker container."""
        assert self._session is not None
        logger.info(f"Calling tool: {tool_name}")
        logger.debug(f"Tool arguments: {tool_args}")
        result = await self._session.call_tool(tool_name, tool_args)
        logger.debug(f"Tool {tool_name} executed successfully")
        return result

    async def setup(self) -> CallToolResult:
        """Setup the Docker container."""
        assert self.setup_tool is not None
        logger.info(f"Running setup tool: {self.setup_tool}")
        logger.debug(f"Setup arguments: {self.setup_tool_args}")
        result = await self.call_tool(self.setup_tool, self.setup_tool_args or {})
        logger.info("Setup completed successfully")
        return result

    async def evaluate(self) -> CallToolResult:
        """Evaluate the Docker container."""
        assert self.evaluate_tool is not None
        logger.info(f"Running evaluate tool: {self.evaluate_tool}")
        logger.debug(f"Evaluate arguments: {self.evaluate_tool_args}")
        result = await self.call_tool(self.evaluate_tool, self.evaluate_tool_args or {})
        logger.info("Evaluation completed successfully")
        return result

    @override
    async def __aenter__(self):
        """Async context manager entry."""
        logger.debug("Entering Environment context manager")
        await self.initialize()
        return self

    @override
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """Async context manager exit."""
        logger.debug(f"Exiting Environment context manager (exception: {exc_type})")
        await self.cleanup()
