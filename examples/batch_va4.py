#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch URL Processor for VA4

This script processes multiple URLs from a file or stdin and generates
a report of which programs are relevant and which are not.

Usage:
    python batch_va4.py urls.txt
    cat urls.txt | python batch_va4.py -
    python batch_va4.py --interactive
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_axe.va4_product_discoverer import process_url, ALL_CATEGORIES
from llm_axe.models import OllamaChat

def log(msg: str, verbose: bool = True):
    """Print message if verbose is enabled."""
    if verbose:
        print(msg, flush=True)

def read_urls_from_file(file_path: str) -> List[str]:
    """Read URLs from a file (one URL per line)."""
    urls = []
    
    if file_path == '-':
        # Read from stdin
        for line in sys.stdin:
            url = line.strip()
            if url and not url.startswith('#'):
                urls.append(url)
    else:
        # Read from file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):
                    urls.append(url)
    
    return urls

def process_batch(
    urls: List[str], 
    llm, 
    verbose: bool = True,
    continue_on_error: bool = True
) -> List[Dict]:
    """Process a batch of URLs and return results."""
    
    results = []
    total = len(urls)
    
    log(f"\n{'='*70}", verbose)
    log(f"BATCH PROCESSING: {total} URLs", verbose)
    log(f"{'='*70}\n", verbose)
    
    for i, url in enumerate(urls, 1):
        log(f"[{i}/{total}] Processing: {url}", verbose)
        
        try:
            # Process URL without interactive Q&A
            extracted_data, classification = process_url(
                url, 
                llm, 
                enable_qa=False
            )
            
            result = {
                'index': i,
                'url': url,
                'status': 'success',
                'programme_name': extracted_data.get('programme_name', 'N/A'),
                'category': classification.get('primary_category', 'unknown'),
                'category_display': ALL_CATEGORIES.get(
                    classification.get('primary_category', 'other'),
                    'Unknown'
                ),
                'is_relevant': classification.get('is_relevant', False),
                'confidence': classification.get('confidence', 0.0),
                'reasoning': classification.get('reasoning', ''),
                'key_features': classification.get('key_features', [])
            }
            
            results.append(result)
            
            status = "✓ RELEVANT" if result['is_relevant'] else "✗ NOT RELEVANT"
            log(f"  → {status} ({result['category']}, {result['confidence']:.0%})", verbose)
            
        except Exception as e:
            log(f"  ✗ ERROR: {e}", verbose)
            
            result = {
                'index': i,
                'url': url,
                'status': 'error',
                'error': str(e),
                'programme_name': 'ERROR',
                'category': 'error',
                'category_display': 'Error',
                'is_relevant': False,
                'confidence': 0.0,
                'reasoning': '',
                'key_features': []
            }
            
            results.append(result)
            
            if not continue_on_error:
                log("\n[ERROR] Stopping due to error (use --continue-on-error to skip)", verbose)
                break
        
        log("", verbose)  # Empty line
    
    return results

def generate_report(results: List[Dict], output_format: str = 'text') -> str:
    """Generate a report from batch processing results."""
    
    if output_format == 'json':
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    # Text report
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("VA4 BATCH PROCESSING REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report_lines.append(f"Total URLs: {len(results)}")
    
    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    relevant = [r for r in results if r['is_relevant']]
    
    report_lines.append(f"Successful: {len(successful)}")
    report_lines.append(f"Errors: {len(errors)}")
    report_lines.append(f"Relevant: {len(relevant)}")
    report_lines.append("")
    
    # Summary table
    report_lines.append("="*70)
    report_lines.append("DETAILED RESULTS")
    report_lines.append("="*70)
    report_lines.append(f"{'#':<4} {'Status':<10} {'Category':<25} {'Conf':<6} {'Program'}")
    report_lines.append("-"*70)
    
    for r in results:
        if r['status'] == 'success':
            status = "✓ REL" if r['is_relevant'] else "✗ NREL"
            category = r['category'][:24]
            conf = f"{r['confidence']:.0%}"
            program = r['programme_name'][:35]
            report_lines.append(f"{r['index']:<4} {status:<10} {category:<25} {conf:<6} {program}")
        else:
            report_lines.append(f"{r['index']:<4} {'ERROR':<10} {'error':<25} {'N/A':<6} ERROR")
    
    report_lines.append("="*70)
    
    # Categories breakdown
    if relevant:
        report_lines.append("")
        report_lines.append("RELEVANT PROGRAMS BY CATEGORY")
        report_lines.append("-"*70)
        
        # Group by category
        by_category = {}
        for r in relevant:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)
        
        for cat, items in sorted(by_category.items()):
            cat_display = ALL_CATEGORIES.get(cat, cat)
            report_lines.append(f"\n{cat_display} ({len(items)} programs):")
            for item in items:
                report_lines.append(f"  • {item['programme_name']}")
                report_lines.append(f"    URL: {item['url']}")
                if item['key_features']:
                    report_lines.append(f"    Features: {', '.join(item['key_features'][:3])}")
    
    # Errors
    if errors:
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("ERRORS")
        report_lines.append("-"*70)
        for err in errors:
            report_lines.append(f"{err['index']}. {err['url']}")
            report_lines.append(f"   Error: {err.get('error', 'Unknown error')}")
            report_lines.append("")
    
    report_lines.append("="*70)
    
    return "\n".join(report_lines)

def save_report(report: str, output_file: str):
    """Save report to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch URL processor for VA4 Product Discoverer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process URLs from file
  python batch_va4.py urls.txt
  
  # Read from stdin
  cat urls.txt | python batch_va4.py -
  
  # Save report to file
  python batch_va4.py urls.txt -o report.txt
  
  # Generate JSON output
  python batch_va4.py urls.txt --format json -o results.json
  
  # Stop on first error
  python batch_va4.py urls.txt --stop-on-error
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input file with URLs (one per line) or "-" for stdin'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for report (default: print to stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--model',
        default='deepseek-r1:latest',
        help='Ollama model to use (default: deepseek-r1:latest)'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop processing on first error'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode: enter URLs manually'
    )
    
    args = parser.parse_args()
    
    # Get URLs
    if args.interactive:
        print("Enter URLs (one per line, empty line to finish):")
        urls = []
        while True:
            try:
                url = input("> ").strip()
                if not url:
                    break
                if url and not url.startswith('#'):
                    urls.append(url)
            except (EOFError, KeyboardInterrupt):
                print()
                break
    elif args.input:
        urls = read_urls_from_file(args.input)
    else:
        parser.print_help()
        sys.exit(1)
    
    if not urls:
        print("No URLs to process")
        sys.exit(1)
    
    # Initialize LLM
    log(f"Initializing LLM ({args.model})...", not args.quiet)
    llm = OllamaChat(model=args.model)
    
    # Process URLs
    results = process_batch(
        urls,
        llm,
        verbose=not args.quiet,
        continue_on_error=not args.stop_on_error
    )
    
    # Generate report
    report = generate_report(results, output_format=args.format)
    
    # Output report
    if args.output:
        save_report(report, args.output)
        # Print summary even in quiet mode
        if args.quiet:
            relevant = sum(1 for r in results if r['is_relevant'])
            print(f"\nProcessed {len(results)} URLs, found {relevant} relevant programs")
    else:
        print(report)

if __name__ == "__main__":
    main()
