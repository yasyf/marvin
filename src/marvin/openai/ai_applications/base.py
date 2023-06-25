import asyncio
import inspect
import math
from enum import Enum
from typing import Any, Union
from uuid import UUID

import prefect
import prefect.states
import uvicorn
from fastapi import Body, FastAPI
from jsonpatch import JsonPatch
from pydantic import BaseModel, Field, PrivateAttr, validator

import marvin
from marvin.bot.history import History, InMemoryHistory
from marvin.models.threads import BaseMessage, Message
from marvin.openai import Tool
from marvin.utilities import prefect_utils
from marvin.utilities.openai import OpenAIFunction, call_llm_chat
from marvin.utilities.strings import jinja_env
from marvin.utilities.types import LoggerMixin, MarvinBaseModel


class ApplicationResponse(BaseMessage):
    parsed_content: Any = None


AI_APP_SYSTEM_MESSAGE = jinja_env.from_string(inspect.cleandoc("""
    # Overview
    
    You are the intelligent, natural language interface to an application. The
    application has a structured `state` but no formal API; you are the only way
    to interact with it. You must interpret the user's inputs as attempts to
    interact with the application's state in the context of the application's
    purpose. For example, if the application is a to-do tracker, then "I need to
    go to the store" should be interpreted as an attempt to add a new to-do
    item. If it is a route planner, then "I need to go to the store" should be
    interpreted as an attempt to find a route to the store. 
    
    # Instructions
    
    Your primary job is to maintain the application's `state` and your own
    `ai_state`. Together, these two states fully parameterize the application,
    making it resilient, serializable, and observable. You do this autonomously;
    you do not need to inform the user of any changes you make. 
    
    # Actions
    
    Each time the user runs the application by sending a message, you must take
    the following steps:
    
    - Call the `UpdateAIState` function to update your own state. Use your state
      to track notes, objectives, in-progress work, and to break problems down
      into solvable, possibly dependent parts. You state consists of a few
      fields:
        - `notes`: a list of notes you have taken. Notes are free-form text and
          can be used to track anything you want to remember, such as
          long-standing user instructions, or observations about how to behave
          or operate the application. These are exclusively related to your role
          as intermediary and you interact with the user and application. Do not
          track application data or state here.
        - `tasks`: a list of tasks you are working on. Tasks form a DAG and
          track goals, milestones, in-progress work, or break problems down into
          all the discrete steps needed to solve them. You should create a new
          task for EITHER any work that will require a function call other than
          updating state OR will require more than one state update to complete.
          You should also create a new task for any work or action you expect to
          take in the future; you can always cancel it if you don't. You do not
          need to create tasks for simple state updates. Update task states
          appropriately. 
            - use `upstream_task_ids` to indicate that a task can not start
              until the upstream tasks are completed.
            - use `nest_task_id` to indicate that this task must be completed
              before the nest task can be completed.

    - Call any functions necessary to achieve the application's purpose.
    
    - Call the `UpdateAppState` function to update the application's state. This
      is where you should store any information relevant to the application
      itself.

    You can call these functions at any time, in any order, as necessary.
    Finally, respond to the user with an informative message. Remember that the
    user is probably uninterested in the internal steps you took, so respond
    only in a manner appropriate to the application's purpose.

    # Current details
    
    ## Application description
     
    {{ app.description }}
    
    ## Application state
     
    {{ app.state.state.json() }}
    
    with schema
    
    {{app.state.state.schema()}}
    
    ## AI (your) state
    
    {{ app.ai_state.json() }} 
    
    with schema
    
    {{app.ai_state.schema()}}
    
    ## Today's date
    
    {{ dt() }}
    
    """))


class TaskState(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Task(BaseModel):
    class Config:
        validate_assignment = True

    id: int
    name: str
    description: str = None
    upstream_task_ids: list[int] = None
    nest_task_id: int = None
    state: TaskState = TaskState.PENDING
    prefect_flow_run_id: UUID = None


class AIState(BaseModel):
    tasks: list[Task] = []
    notes: list[str] = []


class FreeformState(BaseModel):
    state: dict[str, Any] = {}


class AppState(BaseModel):
    state: BaseModel = Field(default_factory=FreeformState)
    prefect_flow_run_id: UUID = None


class AIApplication(MarvinBaseModel, LoggerMixin):
    name: str = None
    description: str
    state: AppState = Field(default_factory=AppState)
    ai_state: AIState = Field(default_factory=AIState)
    tools: list[Tool] = []
    history: History = Field(default_factory=InMemoryHistory)

    @validator("name", pre=True, always=True)
    def validate_name(cls, v):
        if v is None:
            v = cls.__name__
        return v

    @validator("state", pre=True, always=True)
    def validate_state(cls, v):
        # users will often pass dicts or initial state objects, but we need
        # to embed them in an `AppState` object
        if v is None:
            v = AppState()
        elif not isinstance(v, AppState):
            try:
                type(v)()
            except Exception:
                raise ValueError(
                    "The provided state object can not be initialized with no data, so"
                    " it can not be cleared. Please provide a state object that can be"
                    " cleared."
                )
            v = AppState(state=v)
        return v

    @validator("tools", pre=True)
    def validate_tools(cls, v):
        v = [t.as_tool() if isinstance(t, AIApplication) else t for t in v]
        return v

    async def clear_state(self):
        if self.state.prefect_flow_run_id:
            async with prefect.get_client() as client:
                client.set_flow_run_state(
                    flow_run_id=self.state.prefect_flow_run_id,
                    state=prefect.states.Completed(),
                )
            self.state.prefect_flow_run_id = None

        cleared_state = type(self.state.state)()
        self.state = AppState(state=cleared_state)
        self.ai_state = AIState()

    async def run(self, input_text: str = None):
        # if we don't have a flow run id for this invocation, create one
        if not self.state.prefect_flow_run_id:
            flow_run = await prefect_utils.create_flow_run(
                flow_name=self.name,
                state=prefect.states.Pending(),
            )
            self.state.prefect_flow_run_id = flow_run.id

        return await self._run(input_text=input_text)

    async def _run(self, input_text: str = None):
        # put a placeholder for the system message
        messages = [None]

        historical_messages = await self.history.get_messages()
        messages.extend(historical_messages)

        if input_text:
            self.logger.debug_kv("User input", input_text, key_style="green")
            input_message = Message(role="user", content=input_text)
            await self.history.add_message(input_message)
            messages.append(input_message)

        # set up tools
        tools = [
            TaskTool(app=self, tool=t)
            for t in [*self.tools, UpdateAppState(app=self), UpdateAIState(app=self)]
        ]

        i = 1
        max_iterations = marvin.settings.llm_max_function_iterations or math.inf
        while i <= max_iterations:
            # always regenerate the system message before calling the LLM
            # so that it will reflect any changes to state
            messages[0] = Message(
                role="system", content=AI_APP_SYSTEM_MESSAGE.render(app=self)
            )

            # every call to the LLM is represented as a task
            call_llm_task = call_llm_chat

            task_run = await prefect_utils.create_task_run(
                flow_run_id=self.state.prefect_flow_run_id,
                task_name="call_llm",
                state=prefect.states.Running(),
                task_inputs=dict(
                    messages=messages, functions=[t.as_openai_function() for t in tools]
                ),
            )

            try:
                response = await call_llm_task(
                    messages=messages,
                    functions=[t.as_openai_function() for t in tools],
                    function_call="auto" if i < max_iterations else "none",
                )
            except Exception as exc:
                await prefect_utils.set_task_run_state(
                    task_run_id=task_run.id,
                    state=prefect.states.Failed(message=str(exc)),
                )
                raise exc
            await prefect_utils.set_task_run_state(
                task_run_id=task_run.id,
                state=prefect.states.Completed(data=response),
            )

            # if the result was a function call, then run the LLM again with the
            # output
            # TODO: find a good way to support short-circuiting execution
            # e.g. raise a END exception
            if response.role == "function":
                messages.append(response)
                i += 1
            else:
                break

        self.logger.debug_kv("AI response", response.content, key_style="blue")
        await self.history.add_message(response)
        return response

    async def serve(self, host="127.0.0.1", port=8000):
        app = FastAPI()

        state_type = type(self.state)

        @app.get("/state")
        def get_state() -> state_type:
            return self.state

        @app.post("/run")
        async def run(text: str = Body(embed=True)) -> ApplicationResponse:
            return await self.run(text)

        config = uvicorn.config.Config(app=app, host=host, port=port)
        server = uvicorn.Server(config=config)
        await server.serve()

    def as_tool(self) -> Tool:
        return AIApplicationTool(app=self)


class AIApplicationTool(Tool):
    app: "AIApplication"

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = type(kwargs["app"]).name
        super().__init__(**kwargs)

    def run(self, input_text: str) -> str:
        return asyncio.run(self.app.run(input_text))


class JSONPatchModel(BaseModel):
    op: str
    path: str
    value: Union[str, float, int, bool, list, dict] = None
    from_: str = Field(None, alias="from")

    class Config:
        allow_population_by_field_name = True


class UpdateAppState(Tool):
    """
    Updates state using JSON Patch documents.
    """

    _app: "AIApplication" = PrivateAttr()
    description = """
        Update the application state by providing a list of JSON patch
        documents. It will be validated against the app state JSON schema.
        """

    def __init__(self, app: AIApplication, **kwargs):
        self._app = app
        super().__init__(**kwargs)

    def run(self, patches: list[JSONPatchModel]):
        patch = JsonPatch(patches)
        updated_state = patch.apply(self._app.state.dict())
        self._app.state = type(self._app.state)(**updated_state)
        return f"Application state updated successfully! (payload was {patches})"


class UpdateAIState(Tool):
    """
    Updates state using JSON Patch documents.
    """

    _app: "AIApplication" = PrivateAttr()
    description = """
        Update the AI state by providing a list of JSON patch documents. It will
        be validated against the AI state JSON schema. Never update a task's
        Prefect flow id.
        """

    def __init__(self, app: AIApplication, **kwargs):
        self._app = app
        super().__init__(**kwargs)

    async def run(self, patches: list[JSONPatchModel]):
        patch = JsonPatch(patches)
        updated_state = patch.apply(self._app.ai_state.dict())

        old_state = self._app.ai_state
        new_state = AIState(**updated_state)

        # update the state of any existing tasks
        for new_task in new_state.tasks:
            old_task = next((t for t in old_state.tasks if t.id == new_task.id), None)

            # the new task was just created
            if old_task is None:
                flow_run = await prefect_utils.create_flow_run(
                    flow_name=f"{new_task.name}",
                    flow_run_name=f"AI plan {new_task.id}",
                    state=prefect.states.Pending(),
                    parent_flow_run_id=self._app.state.prefect_flow_run_id,
                )
                new_task.prefect_flow_run_id = flow_run.id

            # the task has a state update
            elif new_task.state != old_task.state:
                if new_task.state == TaskState.COMPLETED:
                    prefect_state = prefect.states.Completed()
                elif new_task.state == TaskState.FAILED:
                    prefect_state = prefect.states.Failed()
                elif new_task.state == TaskState.IN_PROGRESS:
                    prefect_state = prefect.states.Running()
                elif new_task.state == TaskState.PENDING:
                    prefect_state = prefect.states.Pending()

                await prefect_utils.set_flow_run_state(
                    flow_run_id=new_task.prefect_flow_run_id,
                    state=prefect_state,
                )

        # the task was deleted
        for old_task in old_state.tasks:
            new_task = next((t for t in new_state.tasks if t.id == old_task.id), None)
            if new_task is None and old_task.state not in (
                TaskState.COMPLETED,
                TaskState.FAILED,
            ):
                await prefect_utils.set_flow_run_state(
                    flow_run_id=new_task.prefect_flow_run_id,
                    state=prefect.states.Cancelled(),
                )

        self._app.ai_state = new_state
        return f"AI state updated successfully! (payload was {patches})"


class TaskTool(Tool):
    """
    This tool creates Prefect tasks from any wrapped function call and
    correlates them to an application's task list
    """

    tool: Tool
    _app: "AIApplication" = PrivateAttr()

    def __init__(self, app: AIApplication, **kwargs):
        self._app = app
        super().__init__(**kwargs)

    def as_openai_function(self) -> OpenAIFunction:
        fn_def = self.tool.as_openai_function()
        fn_def.description += (
            "\n\nIf this function call relates to any of your AI state tasks, provide"
            " the task_id_ for tracking purposes."
        )
        fn_def._fn = self.run
        fn_def.parameters["properties"]["task_id_"] = {"type": "integer"}

        return fn_def

    async def run(self, task_id_: int = None, **kwargs):
        app_task = next((t for t in self._app.ai_state.tasks if t.id == task_id_), None)

        if app_task is None:
            prefect_flow_run_id = self._app.state.prefect_flow_run_id
        else:
            prefect_flow_run_id = app_task.prefect_flow_run_id

        task_run = await prefect_utils.create_task_run(
            flow_run_id=prefect_flow_run_id,
            task_name=self.tool.name,
            task_inputs=kwargs,
            state=prefect.states.Running(),
        )
        try:
            result = self.tool.run(**kwargs)
            if inspect.iscoroutine(result):
                result = await result
        except Exception as exc:
            await prefect_utils.set_task_run_state(
                task_run_id=task_run.id, state=prefect.states.Failed(message=str(exc))
            )
            raise exc
        await prefect_utils.set_task_run_state(
            task_run_id=task_run.id, state=prefect.states.Completed(data=result)
        )
        return result
