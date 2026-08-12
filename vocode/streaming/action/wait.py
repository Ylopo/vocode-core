from typing import Type

from loguru import logger
from pydantic.v1 import BaseModel

from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput


class WaitVocodeActionConfig(VocodeActionConfig, type="action_wait"):  # type: ignore
    pass


class WaitParameters(BaseModel):
    pass


class WaitResponse(BaseModel):
    success: bool


class Wait(
    BaseAction[
        WaitVocodeActionConfig,
        WaitParameters,
        WaitResponse,
    ]
):
    description: str = (
        "Use this action to wait for the IVR to finish talking or to continue waiting on hold."
    )
    parameters_type: Type[WaitParameters] = WaitParameters
    response_type: Type[WaitResponse] = WaitResponse

    def __init__(
        self,
        action_config: WaitVocodeActionConfig,
    ):
        super().__init__(
            action_config,
            quiet=True,
            should_respond="never",
        )

    async def run(self, action_input: ActionInput[WaitParameters]) -> ActionOutput[WaitResponse]:
        if await self.wait_for_turn_and_check_interrupted(action_input):
            logger.warning("Triggering message was interrupted")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=WaitResponse(success=False),
            )

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=WaitResponse(success=True),
        )
