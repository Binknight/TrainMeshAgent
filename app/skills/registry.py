"""
Skill Registry — discovers, registers, and dispatches skills.

Features:
  1. Auto-discovery: scans skill directories for SKILL.md + Python modules
  2. Registration: skills register via registry.register(skill_instance)
  3. Tool list generation: registry.list_tools() → OpenAI function schemas
  4. Unified dispatch: registry.execute_tool(name, args, context) → SkillResult
  5. Built-in guardrail + retry loop (per prompt.txt §2.3)
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.models.schemas import GuardrailResult

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent


class SkillRegistry:
    """Central registry for all available skills."""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill instance."""
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' already registered, overwriting")
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if found and removed."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get(self, name: str) -> BaseSkill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[BaseSkill]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_tools(self) -> list[dict]:
        """Generate OpenAI function-calling tool definitions for all skills."""
        return [skill.tool_schema for skill in self._skills.values()]

    def execute_tool(
        self,
        name: str,
        arguments: dict,
        context: SkillContext,
        max_retries: int = 3,
    ) -> SkillResult:
        """
        Unified dispatch with guardrail + retry loop.

        Flow:
          1. Find skill by name
          2. Run input_guard → fail early if invalid
          3. Execute skill
          4. Run output_guard → retry up to max_retries if fails
          5. Return standardized SkillResult
        """
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(success=False, error=f"Unknown skill: {name}")

        # Step 1: Input guardrail
        in_guard = skill.input_guard(arguments)
        if not in_guard.passed:
            return SkillResult(
                success=False,
                error="input_guardrail_failed",
                guardrail_result=in_guard,
            )

        # Step 2: Execute with output guardrail + retry
        last_error = None
        for attempt in range(max_retries):
            result = skill.execute(arguments, context)
            if not result.success:
                # Execution itself failed — don't retry
                result.guardrail_result = None
                return result

            # Step 3: Output guardrail
            out_guard = skill.output_guard(result.data)
            if out_guard.passed:
                result.guardrail_result = out_guard
                return result

            last_error = out_guard
            logger.warning(
                f"Skill '{name}' output guard failed (attempt {attempt + 1}/{max_retries}): "
                f"{out_guard.errors}"
            )

        # Exhausted retries
        return SkillResult(
            success=False,
            error="output_guardrail_failed_after_retries",
            guardrail_result=last_error,
        )

    def discover_from_directory(self, directory: Path | None = None) -> int:
        """
        Auto-discover skills from a directory.

        Scans subdirectories for SKILL.md files and attempts to import
        the corresponding Python module (skill.py or __init__.py)
        that exposes a BaseSkill subclass.
        """
        directory = directory or _SKILLS_DIR
        count = 0

        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue

            # Try to import the Python module
            skill_instance = self._load_skill_from_dir(entry)
            if skill_instance:
                self.register(skill_instance)
                count += 1

        return count

    def _load_skill_from_dir(self, skill_dir: Path) -> BaseSkill | None:
        """Load a skill from a directory containing SKILL.md and a Python module."""
        skill_name = skill_dir.name

        # Read SKILL.md metadata
        metadata = self._parse_skill_md(skill_dir / "SKILL.md")
        if not metadata:
            logger.warning(f"No valid YAML frontmatter in {skill_dir}/SKILL.md")
            return None

        # Try importing the Python module
        module_path = f"app.skills.{skill_name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Try skill.py inside the directory
            alt_path = f"app.skills.{skill_name}.skill"
            try:
                module = importlib.import_module(alt_path)
            except ImportError:
                logger.warning(f"Cannot import Python module for skill '{skill_name}'")
                return None

        # Find BaseSkill subclass in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSkill)
                and attr is not BaseSkill
            ):
                return attr()
            elif isinstance(attr, BaseSkill):
                return attr

        logger.warning(f"No BaseSkill instance found in {module_path}")
        return None

    @staticmethod
    def _parse_skill_md(filepath: Path) -> dict[str, str] | None:
        """Parse YAML frontmatter from a SKILL.md file."""
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return None

        # Extract YAML frontmatter between --- delimiters
        if not content.startswith("---"):
            return None

        end = content.find("---", 3)
        if end == -1:
            return None

        yaml_str = content[3:end].strip()
        try:
            metadata = yaml.safe_load(yaml_str)
            return metadata if isinstance(metadata, dict) else None
        except yaml.YAMLError as e:
            logger.warning(f"YAML parse error in {filepath}: {e}")
            return None


# Singleton
registry = SkillRegistry()
