"""
Skills package — auto-discovers and registers all skills on import.

Skills follow the Anthropic skill directory format:
  skill-name/
    SKILL.md          — YAML frontmatter (name, description) + markdown
    __init__.py       — Python module with BaseSkill subclass

To add a new skill:
  1. Create a directory under app/skills/
  2. Add SKILL.md with YAML frontmatter
  3. Add __init__.py with a BaseSkill subclass
  4. The registry auto-discovers it on startup
"""
from app.skills.registry import registry

# Auto-discover all skills from this directory
from pathlib import Path
_skills_dir = Path(__file__).parent
_discovered = registry.discover_from_directory(_skills_dir)

from app.skills.base import BaseSkill, SkillContext, SkillResult  # noqa: E402, F401
from app.skills.registry import SkillRegistry  # noqa: E402, F401
