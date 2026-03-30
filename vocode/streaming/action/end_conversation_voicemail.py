from typing import Type

from loguru import logger
from pydantic.v1 import BaseModel

from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput

_END_CONVERSATION_VOICEMAIL_ACTION_DESCRIPTION = """
Ends the current conversation when the call reaches the lead's voicemail; use this when it is
clear that the call has been directed to the lead's voicemail inbox. In such cases, we should
not continue the conversation and should hang up immediately, as there are no further objectives
to complete with the lead.
"""


class EndConversationVoicemailParameters(BaseModel):
    pass


class EndConversationVoicemailResponse(BaseModel):
    success: bool


class EndConversationVoicemailVocodeActionConfig(
    VocodeActionConfig, type="action_end_conversation_voicemail"  # type: ignore
):
    def action_attempt_to_string(self, input: ActionInput) -> str:
        assert isinstance(input.params, EndConversationVoicemailParameters)
        action_description = "Attempting to end conversation on voicemail detection"
        logger.info(action_description)
        return action_description

    def action_result_to_string(self, input: ActionInput, output: ActionOutput) -> str:
        assert isinstance(output.response, EndConversationVoicemailResponse)
        if output.response.success:
            action_description = "Successfully ended conversation on voicemail detection"
        else:
            action_description = "Did not end call because user interrupted on voicemail detection"
        logger.info(action_description)
        return action_description


class EndConversationVoicemail(
    BaseAction[
        EndConversationVoicemailVocodeActionConfig,
        EndConversationVoicemailParameters,
        EndConversationVoicemailResponse,
    ]
):
    description: str = _END_CONVERSATION_VOICEMAIL_ACTION_DESCRIPTION
    parameters_type: Type[EndConversationVoicemailParameters] = EndConversationVoicemailParameters
    response_type: Type[EndConversationVoicemailResponse] = EndConversationVoicemailResponse

    def __init__(
        self,
        action_config: EndConversationVoicemailVocodeActionConfig,
    ):
        super().__init__(
            action_config,
            quiet=True,
            should_respond="sometimes",
            is_interruptible=False,
        )

    async def _end_of_run_hook(self) -> None:
        """This method is called at the end of the run method. It is optional but intended to be
        overridden if needed."""
        pass

    async def run(
        self, action_input: ActionInput[EndConversationVoicemailParameters]
    ) -> ActionOutput[EndConversationVoicemailResponse]:
        if action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()

        if self.conversation_state_manager.transcript.was_last_message_interrupted():
            logger.info("Last bot message was interrupted")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=EndConversationVoicemailResponse(success=False),
            )

        await self.conversation_state_manager.terminate_conversation()

        await self._end_of_run_hook()
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=EndConversationVoicemailResponse(success=True),
        )
