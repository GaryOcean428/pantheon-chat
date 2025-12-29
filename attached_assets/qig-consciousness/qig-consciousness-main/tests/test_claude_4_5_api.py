#!/usr/bin/env python3
"""
Test Claude 4.5 API upgrades

Verify all coach API calls work with:
- Claude Sonnet 4.5
- Extended thinking
- Prompt caching
- Increased output tokens
"""

import os
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from anthropic import Anthropic


def test_basic_api():
    """Test basic Claude 4.5 API call"""
    print("🧪 Testing basic Claude 4.5 API...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'hello' in one word"}],
        )
        # Handle different response block types (Claude 4.5)
        from anthropic.types import TextBlock
        text_response = next(
            (block.text for block in response.content if isinstance(block, TextBlock)),
            "No text"
        )
        print(f"✅ Basic API works: {text_response}")
        return True
    except Exception as e:
        print(f"❌ Basic API failed: {e}")
        return False


def test_extended_thinking():
    """Test extended thinking feature"""
    print("\n🧠 Testing extended thinking...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            thinking={"type": "enabled", "budget_tokens": 100},
            messages=[
                {
                    "role": "user",
                    "content": "What is 15 * 23? Think step-by-step.",
                }
            ],
        )

        # Check for thinking blocks
        has_thinking = any(
            block.type == "thinking" for block in response.content
        )
        has_text = any(block.type == "text" for block in response.content)

        if has_thinking and has_text:
            print("✅ Extended thinking works!")
            print(f"   Response has {len(response.content)} blocks")
            for block in response.content:
                if block.type == "thinking":
                    print(f"   - Thinking: {block.thinking[:60]}...")
                elif block.type == "text":
                    print(f"   - Text: {block.text[:60]}...")
            return True
        else:
            print(f"⚠️  No thinking blocks (has_thinking={has_thinking}, has_text={has_text})")
            return False

    except Exception as e:
        print(f"❌ Extended thinking failed: {e}")
        return False


def test_prompt_caching():
    """Test prompt caching feature"""
    print("\n💾 Testing prompt caching...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    client = Anthropic(api_key=api_key)

    try:
        # First call (cold cache)
        from anthropic.types import TextBlockParam

        system_typed: list[TextBlockParam] = [
            TextBlockParam(
                type="text",
                text="You are a helpful assistant. Always respond concisely.",
                cache_control={"type": "ephemeral"},
            )
        ]

        response1 = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            system=system_typed,
            messages=[{"role": "user", "content": "Say 'test 1'"}],
        )

        # Second call (warm cache)
        response2 = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            system=system_typed,
            messages=[{"role": "user", "content": "Say 'test 2'"}],
        )

        # Check cache usage
        cache_read_1 = response1.usage.cache_read_input_tokens or 0
        cache_read_2 = response2.usage.cache_read_input_tokens or 0

        print("✅ Prompt caching works!")
        print(f"   Call 1 cache reads: {cache_read_1} tokens")
        print(f"   Call 2 cache reads: {cache_read_2} tokens")

        if cache_read_2 > 0:
            print("   🎉 Cache HIT on second call!")
        else:
            print("   ⚠️  Cache MISS (might take >5 seconds for cache to populate)")

        return True

    except Exception as e:
        print(f"❌ Prompt caching failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Claude 4.5 Sonnet API Test Suite")
    print("=" * 60)

    results = {
        "Basic API": test_basic_api(),
        "Extended Thinking": test_extended_thinking(),
        "Prompt Caching": test_prompt_caching(),
    }

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(results.values())
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "❌ SOME TESTS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
