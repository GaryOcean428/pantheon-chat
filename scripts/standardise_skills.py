#!/usr/bin/env python3
"""
Standardise SKILL.md files to minimal frontmatter format.

The canonical SKILL.md format per OpenAI Codex and Manus specs:
- name: (required)
- description: (required)

Extended fields (license, compatibility, metadata, allowed-tools) are removed
as they are not part of the standard and add noise.
"""

import os
import re
import sys
from pathlib import Path

SKILLS_DIR = Path("/home/ubuntu/pantheon-chat/skills")
CODEX_SKILLS_DIR = Path("/home/ubuntu/pantheon-chat/.codex/skills")

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content
    
    # Find the closing ---
    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return {}, content
    
    frontmatter_text = content[3:end_match.start() + 3]
    body = content[end_match.end() + 3:]
    
    # Simple YAML parsing for our use case
    frontmatter = {}
    current_key = None
    current_value = []
    
    for line in frontmatter_text.split("\n"):
        # Check for key: value pattern
        match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)$", line)
        if match:
            # Save previous key if exists
            if current_key:
                frontmatter[current_key] = "\n".join(current_value).strip()
            current_key = match.group(1)
            current_value = [match.group(2)] if match.group(2) else []
        elif current_key and line.startswith("  "):
            # Continuation of multiline value
            current_value.append(line)
        elif current_key and not line.strip():
            # Empty line in value
            current_value.append("")
    
    # Save last key
    if current_key:
        frontmatter[current_key] = "\n".join(current_value).strip()
    
    return frontmatter, body


def standardise_frontmatter(frontmatter: dict) -> dict:
    """Keep only name and description fields."""
    return {
        "name": frontmatter.get("name", ""),
        "description": frontmatter.get("description", ""),
    }


def format_frontmatter(frontmatter: dict) -> str:
    """Format frontmatter as YAML."""
    lines = ["---"]
    lines.append(f"name: {frontmatter['name']}")
    lines.append(f"description: {frontmatter['description']}")
    lines.append("---")
    return "\n".join(lines)


def process_skill_file(skill_path: Path, dry_run: bool = False) -> dict:
    """Process a single SKILL.md file."""
    result = {
        "path": str(skill_path),
        "status": "unchanged",
        "removed_fields": [],
    }
    
    content = skill_path.read_text()
    frontmatter, body = parse_frontmatter(content)
    
    if not frontmatter:
        result["status"] = "no_frontmatter"
        return result
    
    # Check for extended fields
    extended_fields = ["license", "compatibility", "metadata", "allowed-tools"]
    removed = [f for f in extended_fields if f in frontmatter]
    
    if not removed:
        result["status"] = "already_minimal"
        return result
    
    result["removed_fields"] = removed
    result["status"] = "standardised"
    
    # Create new content
    new_frontmatter = standardise_frontmatter(frontmatter)
    new_content = format_frontmatter(new_frontmatter) + "\n" + body
    
    if not dry_run:
        skill_path.write_text(new_content)
    
    return result


def main():
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("DRY RUN - No files will be modified\n")
    
    # Find all SKILL.md files
    skill_files = []
    
    # Main skills directory
    if SKILLS_DIR.exists():
        for skill_dir in SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    skill_files.append(skill_md)
    
    # .codex/skills directory
    if CODEX_SKILLS_DIR.exists():
        for skill_dir in CODEX_SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    skill_files.append(skill_md)
    
    # Also check for root-level SKILL.md (empty placeholder)
    root_skill = SKILLS_DIR / "SKILL.md"
    if root_skill.exists():
        skill_files.append(root_skill)
    
    print(f"Found {len(skill_files)} SKILL.md files\n")
    
    results = {
        "standardised": [],
        "already_minimal": [],
        "no_frontmatter": [],
        "unchanged": [],
    }
    
    for skill_path in sorted(skill_files):
        result = process_skill_file(skill_path, dry_run)
        results[result["status"]].append(result)
        
        if result["status"] == "standardised":
            print(f"✅ {skill_path.parent.name}: Removed {result['removed_fields']}")
        elif result["status"] == "already_minimal":
            print(f"✓  {skill_path.parent.name}: Already minimal")
        elif result["status"] == "no_frontmatter":
            print(f"⚠️  {skill_path.parent.name}: No frontmatter found")
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Standardised: {len(results['standardised'])}")
    print(f"  Already minimal: {len(results['already_minimal'])}")
    print(f"  No frontmatter: {len(results['no_frontmatter'])}")
    
    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
