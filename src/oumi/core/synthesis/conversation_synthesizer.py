# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import random
from typing import Any

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.conversation import (
    PLANNER_JSON_SCHEMA,
    Conversation,
    Message,
    Role,
)
from oumi.core.types.tool_call import ToolCall, ToolDefinition, ToolResult
from oumi.environments import GroundingFact
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.synthetic_environment import SyntheticEnvironment
from oumi.environments.utils import describe_grounding_default
from oumi.inference.native_tool_calling import (
    NATIVE_TOOL_CALLING_ENGINES,
    supports_native_tool_calling,
)
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json

_STRAGGLER_NUDGE = (
    "Stop calling tools. Based on the information gathered so far, "
    "provide a final natural-language answer to the user."
)


@dataclasses.dataclass
class PlannerPrompt:
    """A planner-prompt conversation and the augmented sample it was built from."""

    augmented_sample: dict
    conversation: Conversation


@dataclasses.dataclass
class OpeningTurnPrompt:
    """An augmented sample plus its opening-turn generation prompt."""

    augmented_sample: dict
    conversation: Conversation


@dataclasses.dataclass
class SeedConversation:
    """A seed conversation plus the ``generation_state`` a turn driver needs."""

    conversation: Conversation
    generation_state: dict


class ConversationSynthesizer:
    """Synthesizes a conversation.

    Args:
        params: The parameters for the conversation synthesizer.
        inference_config: The configuration for the inference engine.
    """

    def __init__(
        self,
        params: GeneralSynthesisParams,
        inference_config: InferenceConfig,
        environment_config: EnvironmentConfig | None = None,
    ):
        """Initialize the synthesizer."""
        self._params = params
        self._environment_config = environment_config
        self._formatter = AttributeFormatter(params)

        self._inference_engine = build_inference_engine(
            engine_type=inference_config.engine or InferenceEngineType.NATIVE,
            model_params=inference_config.model,
            remote_params=inference_config.remote_params,
        )
        self._inference_config = inference_config
        self._default_turn_order = [Role.USER, Role.ASSISTANT]
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cached_tokens: int = 0

        if (
            self._environment_config is not None
            and self._environment_config.all_tools
            and not supports_native_tool_calling(inference_config.engine)
        ):
            supported = sorted(e.value for e in NATIVE_TOOL_CALLING_ENGINES)
            raise ValueError(
                f"Tool synthesis requires an engine with native tool-calling "
                f"support. Configured engine '{inference_config.engine}' does "
                f"not support it. Use one of: {supported}."
            )

        self._router: ToolRouter | None = None
        if self._environment_config is not None:
            self._router = ToolRouter.from_environment_config(
                self._environment_config,
                on_env_built=self._wire_inference,
            )

        self._sample_routers: list[ToolRouter | None] = []

    def _prepare_sample_routers(self, n_samples: int) -> None:
        """Replace ``self._sample_routers`` with one router clone per sample.

        Each sample's tool dispatch and grounding read hit an env with state
        independent of every other sample's. Callers must pair this with
        ``_close_sample_routers`` to release the per-sample envs.
        """
        self._sample_routers = (
            [self._router.for_sample() for _ in range(n_samples)]
            if self._router is not None
            else [None] * n_samples
        )

    def _close_sample_routers(self, *, suppress_errors: bool) -> None:
        """Close each per-sample router, guarding so one failure can't leak the rest.

        Clears the list first, then closes each router; re-raises the first close
        error unless ``suppress_errors`` (set when a body exception is already
        propagating and must not be masked).
        """
        routers, self._sample_routers = self._sample_routers, []
        first_error: BaseException | None = None
        for router in routers:
            if router is None:
                continue
            try:
                router.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

    def _wire_inference(self, env: BaseEnvironment) -> None:
        """Inject the synthesizer's engine + base config into synthetic envs."""
        if isinstance(env, SyntheticEnvironment):
            env.attach_inference(self._inference_engine, self._inference_config)

    def _resolve_available_tools(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> list[ToolParams]:
        """Resolve tools for a multiturn attribute from selected environments."""
        if self._environment_config is None:
            return []
        return self._environment_config.resolve_tools(
            environment_ids=multiturn_attribute.available_environments or None,
            tool_ids=multiturn_attribute.available_tools or None,
        )

    @staticmethod
    def _tool_error(tool_call: ToolCall, msg: str) -> Message:
        return Message(
            role=Role.TOOL,
            tool_call_id=tool_call.id,
            content=json.dumps({"error": msg}),
        )

    @staticmethod
    def _final_assistant_text(msg: Message | None) -> str:
        if msg is None or not isinstance(msg.content, str):
            return ""
        return msg.content

    @staticmethod
    def _tool_message(tool_call: ToolCall, result: ToolResult) -> Message:
        content = (
            result.output
            if isinstance(result.output, str)
            else json.dumps(result.output)
        )
        return Message(
            role=Role.TOOL,
            tool_call_id=tool_call.id,
            content=content,
        )

    def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall], sample_idx: int
    ) -> list[Message]:
        """Dispatch a batch of tool calls; returns one TOOL message per call.

        Validates each call via the router, then groups surviving calls by env
        and routes each group in one batched ``env.step()``. If the batched
        route raises, falls back to per-call routing so individual errors stay
        attributed.

        ``sample_idx`` selects the per-sample router clone built at
        ``synthesize()`` entry; routing through it keeps state mutations
        scoped to one sample's env instances.
        """
        router = self._sample_routers[sample_idx]
        assert router is not None, "tool calls require an environment_config"
        results: list[Message | None] = [None] * len(tool_calls)
        groups: dict[int, list[tuple[int, ToolCall, dict[str, Any]]]] = {}
        for idx, tc in enumerate(tool_calls):
            try:
                arguments = router.parse_and_validate_arguments(
                    tc.function.name, tc.function.arguments
                )
            except (ToolArgumentError, ToolLookupError) as exc:
                results[idx] = self._tool_error(tc, str(exc))
                continue
            env = router.tool_to_env[tc.function.name]
            groups.setdefault(id(env), []).append((idx, tc, arguments))

        for group in groups.values():
            calls = [(tc.function.name, args) for _, tc, args in group]
            try:
                outputs = router.route_batch(calls)
            except Exception:
                # On batch failure, re-route each call individually so per-call
                # errors stay attributed. SyntheticEnvironment's in-batch cache
                # shields earlier successes from re-inference, but calls past
                # the failing index re-infer. Acceptable for attribution today;
                # Phase 2's corrective-retry should replace this fallback.
                for idx, tc, args in group:
                    try:
                        [single] = router.route_batch([(tc.function.name, args)])
                    except Exception as exc:
                        results[idx] = self._tool_error(
                            tc, f"Tool '{tc.function.name}' raised: {exc}"
                        )
                        continue
                    results[idx] = self._tool_message(tc, single)
                continue
            for (idx, tc, _), out in zip(group, outputs):
                results[idx] = self._tool_message(tc, out)

        assert all(r is not None for r in results), "every call must produce a message"
        return results  # type: ignore[return-value]

    def _validate_roles(self, multiturn_attribute: MultiTurnAttribute) -> None:
        """Validate that required roles have corresponding personas.

        Args:
            multiturn_attribute: The multi-turn attribute to validate.

        Raises:
            ValueError: If a required role is missing from role_instruction_messages.
        """
        available_roles = set(multiturn_attribute.role_instruction_messages.keys())

        for role in self._default_turn_order:
            if role not in available_roles:
                raise ValueError(
                    f"Role '{role.value}' is missing from "
                    f"role_instruction_messages. Available roles: "
                    f"{[r.value for r in available_roles]}"
                )

    def synthesize(
        self,
        samples: list[dict],
        multiturn_attributes: MultiTurnAttribute,
    ) -> list[dict[str, dict | str] | None]:
        """Synthesize a multi-turn conversation.

        Order will be identical to the order of the samples.

        Args:
            samples: The samples to synthesize values for.
            multiturn_attributes: The multi-turn attribute defining conversation rules.

        Returns:
            A list aligned to the input samples. Each entry is either:
            - a dictionary containing the conversation and plan, or
            - None when the synthesized conversation is filtered out.
        """
        if not samples:
            return []

        self._validate_roles(multiturn_attributes)

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )
        available_tools = self._resolve_available_tools(multiturn_attributes)
        if available_tools:
            logger.debug(
                "Resolved tools for '%s': %s",
                multiturn_attributes.id,
                [tool.id for tool in available_tools],
            )

        self._prepare_sample_routers(len(samples))
        try:
            self._warn_on_grounding_placeholder(multiturn_attributes)
            self._attach_grounding_facts(samples, multiturn_attributes)
            samples = self._plan_samples(samples, multiturn_attributes)
            conversations = self._synthesize_all_samples(samples, multiturn_attributes)
        except BaseException:
            self._close_sample_routers(suppress_errors=True)
            raise
        else:
            self._close_sample_routers(suppress_errors=False)

        records: list[dict[str, dict | str] | None] = []
        plan_key = f"{multiturn_attributes.id}_plan"
        filtered_count = 0
        for sample, conversation in zip(samples, conversations):
            if self._has_empty_messages(conversation):
                filtered_count += 1
                records.append(None)
                continue
            record: dict[str, dict | str] = {
                multiturn_attributes.id: conversation.to_dict(),
                plan_key: sample["conversation_plan"],
            }
            records.append(record)

        if filtered_count > 0:
            logger.warning(
                f"Filtered out {filtered_count} conversation(s) with empty messages "
                f"out of {len(conversations)} total"
            )

        return records

    def build_planner_prompts(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[PlannerPrompt]:
        """Attach grounding and build planner prompts, without inference.

        Self-contained inference-free entrypoint for callers that drive the
        planner as a separate stage: prepares routers, attaches grounding, and
        renders one prompt per sample. ``target_turns`` is drawn randomly, so
        persist it if plans are inferred out-of-process.

        Args:
            samples: The samples to plan conversations for.
            multiturn_attribute: The multi-turn attribute defining conversation
                rules.

        Returns:
            One :class:`PlannerPrompt` per input sample, in order.
        """
        self._validate_roles(multiturn_attribute)
        self._prepare_sample_routers(len(samples))
        try:
            self._warn_on_grounding_placeholder(multiturn_attribute)
            self._attach_grounding_facts(samples, multiturn_attribute)
            prompts = self._render_planner_prompts(samples, multiturn_attribute)
        except BaseException:
            self._close_sample_routers(suppress_errors=True)
            raise
        else:
            self._close_sample_routers(suppress_errors=False)
            return prompts

    def _render_planner_prompts(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[PlannerPrompt]:
        """Render planner prompts, assuming grounding attachment has already run."""
        turn_order = self._default_turn_order
        prompts: list[PlannerPrompt] = []
        for sample in samples:
            target_turns = self._select_target_turns(multiturn_attribute, turn_order)
            augmented_sample = {
                **sample,
                "target_turns": target_turns,
                "conversation_plan": "",
                "parsed_turn_plans": [""] * target_turns,
            }
            logger.debug(f"Planning conversation with {target_turns} turns")
            prompts.append(
                PlannerPrompt(
                    augmented_sample=augmented_sample,
                    conversation=self._create_planner_prompt(
                        multiturn_attribute, augmented_sample
                    ),
                )
            )
        return prompts

    def build_opening_turn_prompts(
        self,
        samples: list[dict],
        plans: list[str],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[OpeningTurnPrompt]:
        """Parse plans and build the opening (turn 1, USER) generation prompts.

        Inference-free stage after :meth:`build_planner_prompts`: parses each
        plan into per-turn instructions and renders the opening-turn prompt.

        Args:
            samples: Augmented samples from ``build_planner_prompts`` (each
                carrying ``target_turns``).
            plans: Raw planner output strings, aligned 1:1 with ``samples``.
            multiturn_attribute: The multi-turn attribute defining conversation
                rules.

        Returns:
            One :class:`OpeningTurnPrompt` per sample, in order.
        """
        self._validate_roles(multiturn_attribute)
        opening_role = self._default_turn_order[0]
        prompts: list[OpeningTurnPrompt] = []
        for sample, plan in zip(samples, plans):
            target_turns = sample["target_turns"]
            parsed = self._parse_plan(plan, target_turns) or [""] * target_turns
            augmented = {
                **sample,
                "conversation_plan": plan,
                "parsed_turn_plans": parsed,
            }
            prompts.append(
                OpeningTurnPrompt(
                    augmented_sample=augmented,
                    conversation=self._build_turn_prompt(
                        augmented,
                        multiturn_attribute,
                        opening_role,
                        current_turn=1,
                        history=[],
                    ),
                )
            )
        return prompts

    def build_seed_conversations(
        self,
        samples: list[dict],
        opening_turns: list[str],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[SeedConversation]:
        """Build seed conversations for out-of-process multi-turn generation.

        Inference-free stage after :meth:`build_opening_turn_prompts`: pairs each
        opening user utterance with its seed conversation and generation state.

        Args:
            samples: Augmented samples (carrying ``target_turns`` and
                ``parsed_turn_plans``) from ``build_opening_turn_prompts``.
            opening_turns: The opening user utterances, aligned 1:1 with
                ``samples``.
            multiturn_attribute: The multi-turn attribute defining conversation
                rules.

        Returns:
            One :class:`SeedConversation` per sample, in order.
        """
        self._validate_roles(multiturn_attribute)
        assistant_persona = multiturn_attribute.role_instruction_messages[
            Role.ASSISTANT
        ]
        user_persona = multiturn_attribute.role_instruction_messages[Role.USER]
        seeds: list[SeedConversation] = []
        for sample, opening in zip(samples, opening_turns):
            # Personas may reference {current_turn}; the seed is turn 1, matching
            # build_opening_turn_prompts. Rendered once and reused for every turn.
            sample_with_turn = {**sample, "current_turn": 1}
            seed = Conversation(
                messages=[
                    self._format_persona(
                        sample_with_turn, assistant_persona, Role.ASSISTANT
                    ),
                    Message(role=Role.USER, content=opening),
                ]
            )
            output_message = self._format_output_system_message(
                sample, multiturn_attribute.output_system_prompt
            )
            output_system_prompt = (
                output_message.content if output_message is not None else None
            )
            generation_state = {
                "target_turns": sample["target_turns"],
                "turn_plans": sample.get("parsed_turn_plans", []),
                "user_persona": self._formatter.format(
                    sample_with_turn, user_persona, missing_values_allowed=False
                ),
                "output_system_prompt": output_system_prompt,
            }
            seeds.append(
                SeedConversation(conversation=seed, generation_state=generation_state)
            )
        return seeds

    def _plan_samples(
        self,
        samples: list[dict],
        multiturn_attributes: MultiTurnAttribute,
        max_retries: int = 2,
    ) -> list[dict]:
        """Plan the conversation samples with retry logic for failed parses.

        Args:
            samples: The conversation samples to plan.
            multiturn_attributes: The multi-turn attribute defining conversation rules.
            max_retries: Maximum number of retry attempts for failed plan parsing.

        Returns:
            A list of sample dicts augmented with runtime fields
            (target_turns, conversation_plan, parsed_turn_plans).
        """
        planner_prompts = self._render_planner_prompts(samples, multiturn_attributes)
        augmented_samples = [prompt.augmented_sample for prompt in planner_prompts]
        planner_conversations = [prompt.conversation for prompt in planner_prompts]

        indices_to_process = list(range(len(augmented_samples)))

        for attempt in range(max_retries + 1):
            if not indices_to_process:
                break

            plans = self._generate_plan(
                [planner_conversations[i] for i in indices_to_process]
            )

            failed_indices: list[int] = []
            for idx, plan in zip(indices_to_process, plans):
                augmented_sample = augmented_samples[idx]
                target_turns = augmented_sample["target_turns"]
                parsed = self._parse_plan(plan, target_turns)

                if parsed is not None:
                    augmented_sample["conversation_plan"] = plan
                    augmented_sample["parsed_turn_plans"] = parsed
                else:
                    failed_indices.append(idx)
                    if attempt < max_retries:
                        logger.warning(
                            f"Plan parsing failed for sample {idx}, "
                            f"retrying ({attempt + 1}/{max_retries})"
                        )

            indices_to_process = failed_indices

        if indices_to_process:
            logger.warning(
                f"Failed to parse plans for {len(indices_to_process)} samples "
                f"after {max_retries + 1} attempts, proceeding without plan"
            )

        return augmented_samples

    def _parse_plan(self, plan: str, target_turns: int) -> list[str] | None:
        """Parse a guided-decoded planner output into per-turn instructions.

        Expects the ``{"turns": [{"turn": 1, "instruction": "..."}, ...]}``
        shape enforced by ``PLANNER_JSON_SCHEMA``. Anything else returns
        ``None`` so ``_plan_samples``'s retry loop can re-prompt.

        Args:
            plan: The full plan text from the planner.
            target_turns: Expected number of turns.

        Returns:
            List of instruction strings (one per turn), or None if parsing failed.
        """
        if not plan:
            return None

        wrapped = extract_json(plan, expected_type=dict)
        if not isinstance(wrapped, dict) or not isinstance(wrapped.get("turns"), list):
            return None
        turns = wrapped["turns"]

        result = [""] * target_turns
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            turn_num = turn.get("turn")
            instruction = turn.get("instruction", "")

            if isinstance(turn_num, str):
                try:
                    turn_num = int(turn_num)
                except ValueError:
                    continue
            if isinstance(turn_num, int) and 1 <= turn_num <= target_turns:
                result[turn_num - 1] = str(instruction).strip()

        return result

    @property
    def total_input_tokens(self) -> int:
        """Total input/prompt tokens accumulated across all synthesize() calls."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output/completion tokens accumulated across all synthesize() calls."""
        return self._total_output_tokens

    @property
    def total_cached_tokens(self) -> int:
        """Total cached tokens accumulated across all synthesize() calls."""
        return self._total_cached_tokens

    def _accumulate_token_usage(self, inference_results: list[Conversation]) -> None:
        """Accumulate token usage from inference response metadata."""
        for result in inference_results:
            usage = result.metadata.get("usage", {})
            self._total_input_tokens += usage.get("prompt_tokens", 0)
            self._total_output_tokens += usage.get("completion_tokens", 0)
            self._total_cached_tokens += usage.get("cached_tokens", 0)

    def _extract_response(
        self,
        inference_conversations: list[Conversation],
    ) -> list[str]:
        """Get the inference results from the inference conversations.

        If the inference result is not a string or the conversation is empty,
        an empty string will be returned.
        Strips whitespace to avoid API errors with trailing whitespace.
        """
        results = []
        for inference_result in inference_conversations:
            if not inference_result.messages:
                results.append("")
                continue
            content = inference_result.messages[-1].content
            if isinstance(content, str):
                results.append(content.strip())
            else:
                results.append("")
        return results

    def _has_empty_messages(self, conversation: Conversation) -> bool:
        """Check if any non-system message has empty content.

        Skips system messages (synthesizer-generated) and assistant messages
        with ``tool_calls`` (where ``content`` is legitimately empty).
        """
        for message in conversation.messages:
            if message.role == Role.SYSTEM:
                continue
            if message.role == Role.ASSISTANT and message.tool_calls:
                continue
            if not isinstance(message.content, str) or not message.content.strip():
                return True
        return False

    def _format_persona(self, sample: dict, persona: str, role: Role) -> Message:
        """Format the persona for the sample.

        Args:
            sample: The sample dict containing all attributes.
            persona: The persona string to format.
            role: The role for this persona.

        Returns:
            A Message with the formatted persona as a SYSTEM message.
        """
        formatted_content = self._formatter.format(
            sample,
            persona,
            missing_values_allowed=False,
        )
        return Message(
            role=Role.SYSTEM,
            content=formatted_content,
        )

    def _build_role_context(
        self, sample: dict, multiturn_attribute: MultiTurnAttribute
    ) -> str:
        """Build formatted role context for the planner.

        Formats the persona strings for each role.
        The returned string has curly braces escaped ({{ and }}) so it can be
        safely embedded in another template without causing format errors.
        """
        parts = []
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            formatted = self._formatter.format(
                sample, persona, missing_values_allowed=False
            )
            parts.append(f"[{role.value.upper()}]\n{formatted}")

        result = "\n\n".join(parts)
        return result.replace("{", "{{").replace("}", "}}")

    def _build_turn_order_str(self, turn_order: list[Role], target_turns: int) -> str:
        """Build a string showing which role speaks at each turn.

        Args:
            turn_order: The role sequence that repeats.
            target_turns: Total number of turns.

        Returns:
            A string like "Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, ..."
        """
        parts = []
        for i in range(target_turns):
            role = turn_order[i % len(turn_order)]
            parts.append(f"Turn {i + 1}: {role.value.upper()}")
        return ", ".join(parts)

    def _create_planner_prompt(
        self, multiturn_attribute: MultiTurnAttribute, sample: dict
    ) -> Conversation:
        """Create the planner prompt template with role context and turn order.

        Returns a Conversation with a one-shot example for consistent formatting.
        Pairs with :meth:`_planner_inference_config` to drive guided JSON
        decoding against ``PLANNER_JSON_SCHEMA``.
        """
        role_context = self._build_role_context(sample, multiturn_attribute)
        turn_order = self._default_turn_order
        target_turns = sample["target_turns"]
        turn_order_str = self._build_turn_order_str(turn_order, target_turns)

        system_prompt = (
            "You are a conversation planner. Create conversation outlines "
            "that flow logically from start to finish.\n\n"
            "IMPORTANT: Output your plan as a raw JSON object with a `turns` "
            "array. Do not use markdown or code fences. "
            "Each element of `turns` must have: turn (number) and instruction "
            "(string).\n"
            "Your instructions MUST be specific to the role context provided. "
            "Each turn's instruction should reflect what that specific role "
            "would do at that point in the conversation."
        )

        example_request = (
            "Plan a 4-turn conversation.\n"
            "Turn order: Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, "
            "Turn 4: ASSISTANT\n\n"
            "Role context:\n"
            "[USER]\n"
            "You are a customer who has an issue with a recent order.\n\n"
            "[ASSISTANT]\n"
            "You are a helpful support agent who resolves customer issues.\n\n"
            "Additional instructions: Focus on resolving the order issue "
            "efficiently while maintaining a polite and helpful tone."
        )
        example_response = (
            '{"turns": [\n'
            '  {"turn": 1, "instruction": "Greet support and explain the '
            'issue with the order"},\n'
            '  {"turn": 2, "instruction": "Acknowledge the issue and ask '
            'for order details"},\n'
            '  {"turn": 3, "instruction": "Provide order number and describe '
            'the problem further"},\n'
            '  {"turn": 4, "instruction": "Confirm the issue and offer a '
            'resolution"}\n'
            "]}"
        )

        base_prompt = (
            f"Plan a {target_turns}-turn conversation.\n"
            f"Turn order: {turn_order_str}\n\n"
            "Guidelines:\n"
            "- Each turn should build on the previous turn.\n"
            f"- Pace the conversation naturally for {target_turns} turns.\n"
            "- Focus on what happens, not exact wording.\n"
            "- Instructions MUST be specific to the roles and context provided below.\n"
        )

        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        grounding_facts = sample.get("grounding_facts") or []
        if grounding_facts:
            block = describe_grounding_default(grounding_facts)
            base_prompt += (
                "\nGround this plan in these specific entities:\n"
                f"{block}\n"
                "Grounding rules (role-aware):\n"
                "- USER turn instructions MAY inline concrete identifiers "
                "from the list above (e.g. 'order ORD-4421 is late', "
                "'book B007'). The user persona cannot see this list, so "
                "identifiers the user should mention must be written into "
                "their turn instruction.\n"
                "- Treat each entity's non-identifier fields (e.g. status, "
                "due_date, return_date) as preconditions. If a field's value "
                "contradicts the conversation intent -- for example trying "
                "to borrow a book whose status is 'borrowed' or 'overdue', "
                "or trying to return one that is 'available' -- plan a "
                "recovery flow that handles the conflict (offer an "
                "alternative entity from the list, explain the conflict, ask "
                "a clarifying question) instead of a happy-path that the "
                "tool will reject.\n"
                "- ASSISTANT turn instructions MUST NOT pre-resolve or "
                "pre-state any tool output — no identifiers, statuses, "
                "borrower names, due dates, or other facts the assistant "
                "would normally look up. Reference entities by what the "
                "user said (e.g. the title) and describe which TOOL the "
                "assistant should call to resolve or verify. Example — "
                "write 'call lookup_book_status with the book_id from the "
                "catalog', not 'tell the user book B007 is checked out'.\n"
                "- The planner's job for assistant turns is to probe the "
                "right tool usage, not to do the tool's work.\n"
            )

        if multiturn_attribute.conversation_planner:
            formatted_planner = self._formatter.format(
                sample,
                multiturn_attribute.conversation_planner,
                missing_values_allowed=False,
            )
            base_prompt += f"\nAdditional instructions: {formatted_planner}\n"

        base_prompt += "\nOutput ONLY the JSON object. No markdown. No other text."

        return Conversation(
            messages=[
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=example_request),
                Message(role=Role.ASSISTANT, content=example_response),
                Message(role=Role.USER, content=base_prompt),
            ],
        )

    def _generate_plan(self, planners: list[Conversation]) -> list[str]:
        """Generate plans for how the conversations should proceed.

        Args:
            planners: The planner conversation templates (already formatted).

        Returns:
            A list of plan strings, one per sample.
        """
        inference_results = self._inference_engine.infer(
            planners,
            inference_config=self._planner_inference_config(),
        )
        self._accumulate_token_usage(inference_results)

        return self._extract_response(inference_results)

    def _planner_inference_config(self) -> InferenceConfig:
        """Create an inference config for planner calls with JSON guided decoding.

        Returns a copy of ``self._inference_config`` whose ``generation`` block
        carries ``GuidedDecodingParams(json=PLANNER_JSON_SCHEMA)``. The base
        config is left untouched so per-turn (non-planner) inference is not
        constrained.
        """
        return dataclasses.replace(
            self._inference_config,
            generation=dataclasses.replace(
                self._inference_config.generation,
                guided_decoding=GuidedDecodingParams(json=PLANNER_JSON_SCHEMA),
            ),
        )

    def _build_turn_prompt(
        self,
        sample: dict,
        multiturn_attribute: MultiTurnAttribute,
        role: Role,
        current_turn: int,
        history: list[Message],
    ) -> Conversation:
        """Build one turn's generation prompt: persona + history + instruction.

        Shared by the in-process turn loop and out-of-process opening-turn
        generation so both render turns identically.
        """
        target_turns = sample["target_turns"]
        parsed_turn_plans = sample.get("parsed_turn_plans", [])
        turn_idx = current_turn - 1

        turn_instruction = ""
        if 0 <= turn_idx < len(parsed_turn_plans):
            turn_instruction = parsed_turn_plans[turn_idx]

        sample_with_turn = {**sample, "current_turn": current_turn}
        persona = multiturn_attribute.role_instruction_messages[role]
        messages: list[Message] = [
            self._format_persona(sample_with_turn, persona, role)
        ]
        messages.extend(history)

        turn_info = (
            f"You are generating turn {current_turn} of {target_turns} "
            f"as the {role.value.upper()}.\n\n"
        )
        if turn_instruction:
            turn_info += f"For this turn: {turn_instruction}\n\n"
        turn_info += "Generate ONLY your response for this turn. Stay in character."
        messages.append(Message(role=Role.USER, content=turn_info))

        return Conversation(messages=messages)

    def _synthesize_all_samples(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[Conversation]:
        """Synthesize multi-turn conversations for all samples with batched inference.

        Args:
            samples: List of sample dicts with runtime fields (target_turns,
                conversation_plan).
            multiturn_attribute: The multi-turn attribute defining conversation rules.

        Returns:
            List of Conversation objects, one per sample.
        """
        if not samples:
            return []

        histories: list[list[Message]] = [[] for _ in samples]
        max_turns = max(sample["target_turns"] for sample in samples)

        available_tools = self._resolve_available_tools(multiturn_attribute)
        assistant_tools: list[ToolDefinition] | None = (
            [t.to_tool_definition() for t in available_tools]
            if available_tools
            else None
        )

        for turn_idx in range(max_turns):
            current_turn = turn_idx + 1

            prompts: list[Conversation] = []
            sample_indices: list[int] = []
            roles_for_turn: list[Role] = []

            for i, sample in enumerate(samples):
                if turn_idx >= sample["target_turns"]:
                    continue

                turn_order = self._default_turn_order
                role = turn_order[turn_idx % len(turn_order)]
                roles_for_turn.append(role)

                prompts.append(
                    self._build_turn_prompt(
                        sample, multiturn_attribute, role, current_turn, histories[i]
                    )
                )
                sample_indices.append(i)

            if not prompts:
                break

            # roles_for_turn is uniform: role is picked per turn_idx, not per sample.
            uniform_role = roles_for_turn[0]

            if uniform_role == Role.ASSISTANT:
                self._run_assistant_agentic_loop(
                    prompts=prompts,
                    sample_indices=sample_indices,
                    histories=histories,
                    max_consecutive_tool_turns=multiturn_attribute.max_consecutive_tool_turns,
                    assistant_tools=assistant_tools,
                )
                continue

            inference_results = self._inference_engine.infer(
                prompts,
                inference_config=self._inference_config,
            )
            self._accumulate_token_usage(inference_results)

            generated_texts = self._extract_response(inference_results)

            if len(generated_texts) != len(prompts):
                raise RuntimeError(
                    f"Inference engine returned {len(generated_texts)} results "
                    f"but {len(prompts)} prompts were submitted. "
                    f"This may indicate an inference engine error."
                )

            for idx, generated_text, role in zip(
                sample_indices, generated_texts, roles_for_turn
            ):
                histories[idx].append(Message(role=role, content=generated_text))

        conversations: list[Conversation] = []
        for sample, history in zip(samples, histories):
            output_messages: list[Message] = []
            output_message = self._format_output_system_message(
                sample, multiturn_attribute.output_system_prompt
            )
            if output_message:
                output_messages.append(output_message)
            output_messages.extend(history)
            conversations.append(Conversation(messages=output_messages))

        return conversations

    def _run_assistant_tool_round(
        self,
        active: list[int],
        base_msgs: dict[int, list[Message]],
        staging: dict[int, list[Message]],
        round_count: dict[int, int],
        done: dict[int, bool],
        assistant_tools: list[ToolDefinition] | None,
    ) -> None:
        """One round of the assistant→tool agentic loop for the active subset.

        For each active sample: run one assistant inference, then either
        - dispatch the emitted ``tool_calls`` (and bump ``round_count``), or
        - commit the emitted text as the final assistant message and mark ``done``.
        """
        active_prompts = [
            Conversation(
                messages=base_msgs[idx] + staging[idx],
                tools=assistant_tools,
            )
            for idx in active
        ]
        results = self._inference_engine.infer(
            active_prompts,
            inference_config=self._inference_config,
        )
        self._accumulate_token_usage(results)
        if len(results) != len(active):
            raise RuntimeError(
                f"Inference engine returned {len(results)} results for "
                f"{len(active)} prompts in the assistant tool-call loop."
            )
        for idx, result in zip(active, results):
            assistant_msg = result.messages[-1] if result.messages else None
            if assistant_msg is not None and assistant_msg.tool_calls:
                staging[idx].append(
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_msg.content,
                        tool_calls=assistant_msg.tool_calls,
                    )
                )
                staging[idx].extend(
                    self._dispatch_tool_calls(assistant_msg.tool_calls, idx)
                )
                round_count[idx] += 1
            else:
                staging[idx].append(
                    Message(
                        role=Role.ASSISTANT,
                        content=self._final_assistant_text(assistant_msg),
                    )
                )
                done[idx] = True

    def _finalize_stragglers(
        self,
        stragglers: list[int],
        nudged_prompts: list[Conversation],
        staging: dict[int, list[Message]],
    ) -> None:
        """Force a final assistant answer for stragglers (samples at the round cap).

        Each straggler gets one nudged inference (see ``_STRAGGLER_NUDGE``); whatever
        the model emits is committed as the final assistant message. If the model
        defies the nudge and emits more ``tool_calls``, they are preserved on the
        message so it isn't dropped by ``_has_empty_messages`` when ``content`` is None.
        """
        results = self._inference_engine.infer(
            nudged_prompts,
            inference_config=self._inference_config,
        )
        self._accumulate_token_usage(results)
        if len(results) != len(stragglers):
            raise RuntimeError(
                f"Inference engine returned {len(results)} results for "
                f"{len(stragglers)} straggler prompts."
            )
        for idx, result in zip(stragglers, results):
            assistant_msg = result.messages[-1] if result.messages else None
            if assistant_msg is not None and assistant_msg.tool_calls:
                staging[idx].append(
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_msg.content,
                        tool_calls=assistant_msg.tool_calls,
                    )
                )
            else:
                staging[idx].append(
                    Message(
                        role=Role.ASSISTANT,
                        content=self._final_assistant_text(assistant_msg),
                    )
                )

    def _run_assistant_agentic_loop(
        self,
        prompts: list[Conversation],
        sample_indices: list[int],
        histories: list[list[Message]],
        max_consecutive_tool_turns: int,
        assistant_tools: list[ToolDefinition] | None,
    ) -> None:
        """Drive the agentic loop that produces one assistant turn.

        Each iteration calls ``_run_assistant_tool_round`` on the active subset
        (samples that aren't done and haven't hit ``max_consecutive_tool_turns``).
        Samples that exit by hitting the cap are then handled by
        ``_finalize_stragglers`` to force a final answer.

        Mutates ``histories`` in place by extending each per-sample list with the
        assistant + tool messages produced during this loop.
        """
        base_msgs: dict[int, list[Message]] = {
            idx: prompt.messages for idx, prompt in zip(sample_indices, prompts)
        }
        staging: dict[int, list[Message]] = {idx: [] for idx in sample_indices}
        round_count: dict[int, int] = {idx: 0 for idx in sample_indices}
        done: dict[int, bool] = {idx: False for idx in sample_indices}

        while True:
            active = [
                idx
                for idx in sample_indices
                if not done[idx] and round_count[idx] < max_consecutive_tool_turns
            ]
            if not active:
                break

            self._run_assistant_tool_round(
                active=active,
                base_msgs=base_msgs,
                staging=staging,
                round_count=round_count,
                done=done,
                assistant_tools=assistant_tools,
            )

        stragglers = [idx for idx in sample_indices if not done[idx]]
        if stragglers:
            nudge = Message(role=Role.USER, content=_STRAGGLER_NUDGE)
            nudged_prompts = [
                Conversation(messages=base_msgs[idx] + staging[idx] + [nudge])
                for idx in stragglers
            ]
            self._finalize_stragglers(stragglers, nudged_prompts, staging)

        for idx in sample_indices:
            histories[idx].extend(staging[idx])

    def _select_target_turns(
        self, multiturn_attribute: MultiTurnAttribute, turn_order: list[Role]
    ) -> int:
        min_turns = multiturn_attribute.min_turns
        max_turns = multiturn_attribute.max_turns
        target_turns = random.randint(min_turns, max_turns)
        if Role.ASSISTANT not in turn_order:
            return target_turns

        def role_at(turn_count: int) -> Role:
            return turn_order[(turn_count - 1) % len(turn_order)]

        if role_at(target_turns) == Role.ASSISTANT:
            return target_turns
        for turn_count in range(target_turns + 1, max_turns + 1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        for turn_count in range(target_turns - 1, min_turns - 1, -1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        return target_turns

    def _format_output_system_message(
        self,
        sample: dict,
        system_message: str | None,
    ) -> Message | None:
        if system_message is None:
            return None
        formatted_content = self._formatter.format(
            sample,
            system_message,
        )
        return Message(role=Role.SYSTEM, content=formatted_content.strip())

    def _make_grounding_rng(self, seed: int | None, sample_index: int) -> random.Random:
        """Build the per-sample RNG for grounding.

        Seeded mode makes facts deterministic from ``(seed + sample_index)``;
        unseeded uses OS entropy.
        """
        if seed is None:
            return random.Random()
        return random.Random(seed + sample_index)

    def _warn_on_grounding_placeholder(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> None:
        """Warn if ``{grounding_facts}`` appears in user/assistant personas.

        Grounding facts are planner-only — placing the placeholder in a
        user or assistant persona template defeats its purpose and may
        leak env state to roles that should not see it.
        """
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            if not isinstance(persona, str):
                continue
            if "{grounding_facts}" in persona and role in (
                Role.USER,
                Role.ASSISTANT,
            ):
                logger.warning(
                    "MultiTurnAttribute '%s' references {grounding_facts} in "
                    "the %s persona template. grounding is planner-only; "
                    "placing {grounding_facts} in user/assistant templates "
                    "defeats its purpose and may leak env state to roles "
                    "that should not see it.",
                    multiturn_attribute.id,
                    role.value,
                )

    def _attach_grounding_facts(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> None:
        """Attach per-sample grounding facts drawn from grounded envs in scope.

        Writes ``sample["grounding_facts"]`` as a flat list concatenated
        across all envs in scope that declare a ``GroundingConfig``. No-op
        when ``environment_config`` is absent or no env in scope declares
        grounding. Emits one ``logger.warning`` per env when truncation
        occurs (sample_size > pool_size). Each sample reads its grounding
        pool from the same per-sample router that will later receive its
        tool calls.
        """
        if self._environment_config is None:
            return

        scoped_env_ids = (
            set(multiturn_attribute.available_environments)
            if multiturn_attribute.available_environments
            else {env.id for env in self._environment_config.environments}
        )
        grounding_env_params = [
            env_params
            for env_params in self._environment_config.environments
            if env_params.id in scoped_env_ids and env_params.grounding is not None
        ]
        if not grounding_env_params:
            return

        warned_envs: set[str] = set()
        tool_scope = (
            set(multiturn_attribute.available_tools)
            if multiturn_attribute.available_tools
            else None
        )
        for sample_index, sample in enumerate(samples):
            router = self._sample_routers[sample_index]
            assert router is not None, "grounding requires an environment_config"
            facts: list[GroundingFact] = []
            for env_params in grounding_env_params:
                grounding = env_params.grounding
                assert grounding is not None
                env_runtime = router.env_by_id[env_params.id]
                rng = self._make_grounding_rng(grounding.seed, sample_index)
                sampled = env_runtime.sample_grounding(
                    n=grounding.sample_size,
                    rng=rng,
                    tool_ids=tool_scope,
                )
                if (
                    len(sampled) < grounding.sample_size
                    and env_params.id not in warned_envs
                ):
                    logger.warning(
                        "Grounding sample_size=%d exceeds pool size for "
                        "environment '%s'; truncating to %d facts.",
                        grounding.sample_size,
                        env_params.id,
                        len(sampled),
                    )
                    warned_envs.add(env_params.id)
                facts.extend(sampled)
            sample["grounding_facts"] = facts
