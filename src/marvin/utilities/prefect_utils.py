from uuid import UUID

import prefect
import prefect.states


async def create_flow_run(
    flow_name: str,
    flow_run_name: str = None,
    state: prefect.states.State = None,
    parent_flow_run_id: UUID = None,
):
    async with prefect.get_client() as client:
        if parent_flow_run_id:
            parent_task_run = await client.create_task_run(
                task=prefect.Task(name=flow_name, fn=lambda: None),
                flow_run_id=parent_flow_run_id,
                dynamic_key=flow_name,
                state=state or prefect.states.Pending(),
            )
        else:
            parent_task_run = None

        flow_run = await client.create_flow_run(
            name=flow_run_name,
            flow=prefect.Flow(name=flow_name, fn=lambda: None),
            state=state or prefect.states.Pending(),
            parent_task_run_id=parent_task_run.id if parent_flow_run_id else None,
        )

    return flow_run


async def set_flow_run_state(flow_run_id: UUID, state: prefect.states.State):
    async with prefect.get_client() as client:
        await client.set_flow_run_state(flow_run_id=flow_run_id, state=state)


async def create_task_run(
    flow_run_id: UUID,
    task_name: str,
    state: prefect.states.State = None,
    dynamic_key: str = None,
    task_inputs: dict = None,
):
    async with prefect.get_client() as client:
        task_run = await client.create_task_run(
            task=prefect.Task(name=task_name, fn=lambda: None),
            flow_run_id=flow_run_id,
            dynamic_key=dynamic_key or task_name,
            state=state or prefect.states.Pending(),
            # task_inputs=task_inputs,
        )

    return task_run


async def set_task_run_state(task_run_id: UUID, state: prefect.states.State):
    async with prefect.get_client() as client:
        await client.set_task_run_state(task_run_id=task_run_id, state=state)
