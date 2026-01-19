#!/usr/bin/env python3
"""
God Name Resolver CLI Tool
===========================

Command-line tool for cross-mythology god name research and resolution.
Maps external mythology names to Greek canonical archetypes.

Authority: E8 Protocol v4.0, WP5.5
Status: ACTIVE
Created: 2026-01-19

Usage:
    god_name_resolver.py resolve "Odin"
    god_name_resolver.py suggest --domain justice wisdom
    god_name_resolver.py info "Hermes"
    god_name_resolver.py equivalents "Zeus"
    god_name_resolver.py list --mythology norse
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pantheon.cross_mythology import (
    get_cross_mythology_registry,
    resolve_god_name,
    find_similar_gods,
    get_mythology_info,
    get_external_equivalents,
)


def cmd_resolve(args):
    """Resolve external god name to Greek archetype."""
    registry = get_cross_mythology_registry()
    
    try:
        greek_name = registry.resolve_god_name(args.name)
        print(f"\n✓ {args.name} → {greek_name}\n")
        
        if args.verbose:
            info = registry.get_mythology_info(args.name)
            print(f"Mythology: {info['mythology']}")
            print(f"Domain: {', '.join(info['domain'])}")
            print(f"Notes: {info['notes']}")
            if info.get('alternative_mapping'):
                print(f"Alternative mapping: {info['alternative_mapping']}")
    
    except KeyError as e:
        print(f"\n✗ Error: {e}\n")
        
        # Suggest similar names
        available = registry.list_all_external_names()
        similar = [name for name in available if args.name.lower() in name.lower()]
        if similar:
            print(f"Did you mean one of these?")
            for name in similar[:5]:
                print(f"  - {name}")
        print()
        sys.exit(1)


def cmd_suggest(args):
    """Suggest gods by domain keywords."""
    registry = get_cross_mythology_registry()
    
    domains = args.domain
    print(f"\n🔍 Finding gods with domains: {', '.join(domains)}\n")
    
    matches = registry.find_similar_gods(domains)
    
    if not matches:
        print("✗ No gods found matching those domains.\n")
        sys.exit(1)
    
    print("Suggestions (ranked by domain overlap):\n")
    for i, (greek_name, match_count) in enumerate(matches[:args.limit], 1):
        print(f"{i}. {greek_name} ({match_count} domain matches)")
        
        if args.verbose:
            info = registry.get_mythology_info(greek_name)
            if info['is_greek'] and 'external_equivalents' in info:
                equiv = info['external_equivalents']
                equiv_str = ', '.join([
                    f"{myth}: {'/'.join(names)}"
                    for myth, names in list(equiv.items())[:2]
                ])
                print(f"   External equivalents: {equiv_str}")
    
    print()


def cmd_info(args):
    """Get detailed information about a god."""
    registry = get_cross_mythology_registry()
    
    try:
        info = registry.get_mythology_info(args.name)
        
        print(f"\n📖 Information for: {args.name}")
        print("=" * 60)
        
        if info['is_greek']:
            print(f"Type: Greek archetype (canonical name)")
            print(f"Canonical name: {info['canonical_name']}")
            
            if 'external_equivalents' in info:
                print(f"\nExternal equivalents ({info['mapping_count']} mappings):")
                for mythology, names in info['external_equivalents'].items():
                    print(f"  • {mythology.capitalize()}: {', '.join(names)}")
            
            if 'all_domains' in info:
                print(f"\nDomains (across all mythologies):")
                print(f"  {', '.join(info['all_domains'][:20])}")
                if len(info['all_domains']) > 20:
                    print(f"  ... and {len(info['all_domains']) - 20} more")
        
        else:
            print(f"Type: External mythology reference")
            print(f"Mythology: {info['mythology'].capitalize()}")
            print(f"External name: {info['external_name']}")
            print(f"Greek archetype: {info['canonical_name']}")
            print(f"\nDomain: {', '.join(info['domain'])}")
            print(f"\nNotes: {info['notes']}")
            
            if info.get('alternative_mapping'):
                print(f"\nAlternative mapping: {info['alternative_mapping']}")
                print("  (Use for different domain emphasis)")
        
        print()
    
    except KeyError as e:
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


def cmd_equivalents(args):
    """Get external equivalents for a Greek god."""
    registry = get_cross_mythology_registry()
    
    try:
        equivalents = registry.get_external_equivalents(args.greek_name)
        
        if not equivalents:
            print(f"\n✗ No external mappings found for {args.greek_name}\n")
            print("This Greek god may not have documented equivalents in the registry.")
            print()
            sys.exit(1)
        
        print(f"\n🌍 External mythology equivalents for: {args.greek_name}")
        print("=" * 60)
        
        for mythology, names in equivalents.items():
            print(f"\n{mythology.capitalize()}:")
            for name in names:
                print(f"  • {name}")
                
                if args.verbose:
                    info = registry.get_mythology_info(name)
                    print(f"    Domain: {', '.join(info['domain'][:5])}")
        
        print()
    
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


def cmd_list(args):
    """List available god names."""
    registry = get_cross_mythology_registry()
    
    if args.greek:
        names = registry.list_all_greek_archetypes()
        print(f"\n📜 Greek archetypes with external mappings ({len(names)}):")
        print("=" * 60)
        for name in names:
            print(f"  • {name}")
        print()
    
    elif args.mythology:
        names = registry.list_all_external_names(mythology=args.mythology)
        print(f"\n📜 {args.mythology.capitalize()} gods in registry ({len(names)}):")
        print("=" * 60)
        for name in names:
            greek = registry.resolve_god_name(name)
            print(f"  • {name} → {greek}")
        print()
    
    else:
        # List all mythologies
        mythologies = ['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican']
        print(f"\n📜 All external gods in registry:")
        print("=" * 60)
        
        for mythology in mythologies:
            names = registry.list_all_external_names(mythology=mythology)
            print(f"\n{mythology.capitalize()} ({len(names)} gods):")
            for name in names[:5]:
                greek = registry.resolve_god_name(name)
                print(f"  • {name} → {greek}")
            if len(names) > 5:
                print(f"  ... and {len(names) - 5} more")
        
        print()


def cmd_philosophy(args):
    """Display mapping philosophy and design rationale."""
    registry = get_cross_mythology_registry()
    
    philosophy = registry.get_philosophy()
    metadata = registry.get_metadata()
    
    print("\n🏛️  Cross-Mythology Mapping Philosophy")
    print("=" * 60)
    
    print(f"\nCanonical naming: {philosophy.get('canonical_naming', 'Greek')}")
    print(f"\nPurpose:")
    print(f"  {metadata.get('purpose', 'Cross-mythology god name mapping')}")
    
    print(f"\nReasoning:")
    for reason in philosophy.get('reasoning', []):
        print(f"  • {reason}")
    
    print(f"\nUsage notes:")
    for note in philosophy.get('usage', '').split('\n') if isinstance(philosophy.get('usage'), str) else []:
        if note.strip():
            print(f"  • {note.strip()}")
    
    print(f"\nVersion: {metadata.get('version', 'Unknown')}")
    print(f"Authority: {metadata.get('authority', 'Unknown')}")
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="God Name Resolver - Cross-mythology god name mapping tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s resolve "Odin"              # Resolve Odin to Greek archetype
  %(prog)s resolve "Thoth" --verbose   # Resolve with detailed info
  %(prog)s suggest --domain wisdom war # Find gods by domain
  %(prog)s info "Zeus"                 # Get detailed god information
  %(prog)s equivalents "Hermes"        # Get external equivalents
  %(prog)s list --greek                # List all Greek archetypes
  %(prog)s list --mythology norse      # List Norse gods
  %(prog)s philosophy                  # Show mapping philosophy

Authority: E8 Protocol v4.0, WP5.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Resolve command
    resolve_parser = subparsers.add_parser(
        'resolve',
        help='Resolve external god name to Greek archetype'
    )
    resolve_parser.add_argument('name', help='External god name (e.g., "Odin", "Thoth")')
    resolve_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed info')
    
    # Suggest command
    suggest_parser = subparsers.add_parser(
        'suggest',
        help='Suggest gods by domain keywords'
    )
    suggest_parser.add_argument(
        '--domain',
        nargs='+',
        required=True,
        help='Domain keywords (e.g., wisdom war strategy)'
    )
    suggest_parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum number of suggestions (default: 10)'
    )
    suggest_parser.add_argument('-v', '--verbose', action='store_true', help='Show external equivalents')
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Get detailed information about a god'
    )
    info_parser.add_argument('name', help='God name (Greek or external)')
    
    # Equivalents command
    equiv_parser = subparsers.add_parser(
        'equivalents',
        help='Get external equivalents for a Greek god'
    )
    equiv_parser.add_argument('greek_name', help='Greek god name')
    equiv_parser.add_argument('-v', '--verbose', action='store_true', help='Show domain info')
    
    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List available god names'
    )
    list_parser.add_argument('--greek', action='store_true', help='List Greek archetypes')
    list_parser.add_argument(
        '--mythology',
        choices=['egyptian', 'norse', 'hindu', 'sumerian', 'mesoamerican'],
        help='List gods from specific mythology'
    )
    
    # Philosophy command
    phil_parser = subparsers.add_parser(
        'philosophy',
        help='Display mapping philosophy and design rationale'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handlers
    commands = {
        'resolve': cmd_resolve,
        'suggest': cmd_suggest,
        'info': cmd_info,
        'equivalents': cmd_equivalents,
        'list': cmd_list,
        'philosophy': cmd_philosophy,
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
