"""Skill base classes: BaseSkill ABC, SkillContext, SkillResult."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import GuardrailResult


@dataclass
class SkillContext:
    """Dependency injection container passed to every skill execution."""
    session: Any = None          # SessionState
    mcp_client: Any = None       # MCPClient or None
    config: Any = None           # Config object


@dataclass
class SkillResult:
    """Standardized result envelope from any skill execution."""
    success: bool
    data: Any = None
    error: str | None = None
    guardrail_result: GuardrailResult | None = None


class BaseSkill(ABC):
    """
    Abstract base for all TrainMesh Agent skills.

    Each skill is a directory containing:
      SKILL.md  — YAML frontmatter (name, description) + markdown instructions
      module    — Python class extending BaseSkill

    Skills declare their own:
      - Tool schema (for OpenAI function calling)
      - Input guardrail (validate arguments before execution)
      - Output guardrail (validate result after execution)
      - Execute logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier, e.g. 'training-mesh-gen-skill'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""
        ...

    @property
    @abstractmethod
    def tool_schema(self) -> dict:
        """OpenAI function-calling compatible tool definition."""
        ...

    @abstractmethod
    def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        """Execute the skill with given arguments and context."""
        ...

    def input_guard(self, arguments: dict) -> GuardrailResult:
        """Validate input arguments before execution. Override in subclasses."""
        return GuardrailResult(passed=True)

    def output_guard(self, result: Any) -> GuardrailResult:
        """Validate output after execution. Override in subclasses."""
        return GuardrailResult(passed=True)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
