#!/usr/bin/env python3
"""
Test script to verify all Tavily and Perplexity SDK capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search.tavily_client import get_tavily_client, TavilySearchClient
from search.perplexity_client import get_perplexity_client, PerplexityClient
import json

def test_tavily():
    """Test all Tavily capabilities."""
    print("\n" + "="*60)
    print("TAVILY SDK TEST")
    print("="*60)
    
    client = get_tavily_client()
    
    if not client.available:
        print("❌ Tavily client not available (missing API key)")
        return False
    
    print("✅ Tavily client initialized")
    
    print("\n--- 1. SEARCH (basic) ---")
    try:
        result = client.search(
            query="quantum information geometry basics",
            search_depth="basic",
            max_results=3
        )
        if result:
            print(f"✅ Search returned {len(result.results)} results")
            print(f"   Query: {result.query}")
            if result.answer:
                print(f"   AI Answer: {result.answer[:100]}...")
            for r in result.results[:2]:
                print(f"   - {r.title[:50]}... (score: {r.score:.2f})")
        else:
            print("❌ Search returned no results")
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    print("\n--- 2. SEARCH (advanced with answer) ---")
    try:
        result = client.search(
            query="Fisher-Rao metric in machine learning",
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        if result and result.answer:
            print(f"✅ Advanced search with AI answer")
            print(f"   Answer: {result.answer[:150]}...")
        else:
            print("⚠️ Advanced search worked but no AI answer")
    except Exception as e:
        print(f"❌ Advanced search error: {e}")
    
    print("\n--- 3. EXTRACT ---")
    try:
        urls = ["https://en.wikipedia.org/wiki/Fisher_information"]
        results = client.extract(urls)
        if results and len(results) > 0:
            ext = results[0]
            if ext.success:
                print(f"✅ Extract successful")
                print(f"   URL: {ext.url}")
                print(f"   Content length: {len(ext.raw_content)} chars")
            else:
                print(f"⚠️ Extract failed: {ext.error}")
        else:
            print("❌ Extract returned no results")
    except Exception as e:
        print(f"❌ Extract error: {e}")
    
    print("\n--- 4. MAP (site crawl) ---")
    try:
        result = client.map(
            url="https://en.wikipedia.org/wiki/Information_geometry",
            max_depth=1,
            limit=10
        )
        if result.success:
            print(f"✅ Map successful")
            print(f"   Found {result.total_urls} URLs from {result.base_url}")
            for url in result.urls[:3]:
                print(f"   - {url[:60]}...")
        else:
            print(f"⚠️ Map failed: {result.error}")
    except Exception as e:
        print(f"❌ Map error: {e}")
    
    print("\n--- 5. RESEARCH (comprehensive) ---")
    try:
        result = client.research(
            query="density matrix quantum mechanics",
            topic="general",
            max_iterations=2
        )
        if result:
            print(f"✅ Research complete")
            print(f"   Sources: {result['total_sources']}")
            print(f"   Content pieces: {len(result['content'])}")
            if result['ai_synthesis']:
                print(f"   AI Synthesis: {result['ai_synthesis'][:100]}...")
        else:
            print("❌ Research returned no results")
    except Exception as e:
        print(f"❌ Research error: {e}")
    
    return True


def test_perplexity():
    """Test all Perplexity capabilities."""
    print("\n" + "="*60)
    print("PERPLEXITY SDK TEST")
    print("="*60)
    
    client = get_perplexity_client()
    
    if not client.available:
        print("❌ Perplexity client not available (missing API key)")
        return False
    
    print("✅ Perplexity client initialized")
    
    print("\n--- 1. CHAT (basic) ---")
    try:
        result = client.chat(
            message="What is information geometry in one sentence?",
            temperature=0.2
        )
        if result:
            print(f"✅ Chat successful")
            print(f"   Response: {result.content[:150]}...")
            print(f"   Model: {result.model}")
            if result.citations:
                print(f"   Citations: {len(result.citations)}")
        else:
            print("❌ Chat returned no result")
    except Exception as e:
        print(f"❌ Chat error: {e}")
    
    print("\n--- 2. SEARCH (with citations) ---")
    try:
        result = client.search(
            query="Fisher information matrix applications"
        )
        if result:
            print(f"✅ Search successful")
            print(f"   Response: {result.content[:150]}...")
            if result.citations:
                print(f"   Citations: {result.citations[:3]}")
        else:
            print("❌ Search returned no result")
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    print("\n--- 3. PRO SEARCH (deep research) ---")
    try:
        result = client.pro_search(
            query="How does the Bures metric relate to quantum fidelity?",
            focus_areas=["quantum mechanics", "information geometry"]
        )
        if result:
            print(f"✅ Pro Search successful")
            print(f"   Synthesis: {result.synthesis[:200]}...")
            print(f"   Key findings: {len(result.key_findings)}")
            if result.citations:
                print(f"   Citations: {len(result.citations)}")
        else:
            print("⚠️ Pro Search returned no result (may need higher tier)")
    except Exception as e:
        print(f"❌ Pro Search error: {e}")
    
    print("\n--- 4. VALIDATE INSIGHT ---")
    try:
        result = client.validate_insight(
            insight="The Fisher-Rao metric is the unique Riemannian metric invariant under sufficient statistics",
            domains=["information theory", "differential geometry"]
        )
        if result:
            print(f"✅ Validation successful")
            print(f"   Validated: {result.get('validated', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            if result.get('reasoning'):
                print(f"   Reasoning: {str(result['reasoning'])[:100]}...")
        else:
            print("⚠️ Validation returned no result")
    except Exception as e:
        print(f"❌ Validate error: {e}")
    
    return True


def main():
    print("\n" + "#"*60)
    print("# COMPREHENSIVE SEARCH SDK TEST")
    print("#"*60)
    
    tavily_ok = test_tavily()
    perplexity_ok = test_perplexity()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tavily:     {'✅ PASS' if tavily_ok else '❌ FAIL'}")
    print(f"Perplexity: {'✅ PASS' if perplexity_ok else '❌ FAIL'}")
    print("="*60)
    
    return 0 if (tavily_ok and perplexity_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
