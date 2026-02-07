#!/usr/bin/env python
"""
Extract detailed analysis steps from Mathematica notebook.
"""

import re

def main():
    nb_path = r"D:\HEP\ATLAS\LIV\Mathematica code\tag11_nb\tag11.nb"
    
    with open(nb_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print("=" * 70)
    print("DETAILED ANALYSIS PATTERNS IN NOTEBOOK")
    print("=" * 70)
    
    # Search for specific analysis patterns
    patterns = {
        "Signal Injection": r'inject.*signal|signal.*inject|duYZinj|injected',
        "Chi-square / Goodness of fit": r'chi2|chisquare|chi-square|goodness',
        "P-value calculation": r'pvalue|p-value|p value',
        "Confidence intervals": r'confidence|interval|CL\s*=',
        "Upper/Lower limits": r'upper.*limit|lower.*limit|limit.*setting',
        "Pull distribution": r'pull\s*=|pull\[|pullplot',
        "Gaussian/Normal fit": r'gaussian.*fit|normal.*fit|fit.*gaussian',
        "Histogram plots": r'histogram\[|histogramlist',
        "Mean/Average calculation": r'mean\[|average\[',
        "Standard deviation": r'standarddeviation|std\[',
        "Quantile/Percentile": r'quantile\[|percentile',
        "Sensitivity": r'sensitiv',
        "Expected limit": r'expected.*limit|median.*limit',
        "Observed limit": r'observed.*limit',
        "Exclusion": r'exclusion|excluded',
        "Background estimation": r'background|null.*hypothesis',
        "Systematic uncertainty": r'systematic|syst\s*=',
    }
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Get unique matches
            unique = list(set([m.lower()[:30] for m in matches]))[:5]
            print(f"\n{name}: {len(matches)} matches")
            print(f"  Examples: {unique}")
    
    print("\n" + "=" * 70)
    print("SPECIFIC TEXT CELLS MENTIONING KEY ANALYSIS STEPS")
    print("=" * 70)
    
    # Find Text cells with analysis descriptions
    text_pattern = r'Cell\["([^"]{20,200})"[^}]*"Text"'
    text_matches = re.findall(text_pattern, content)
    
    # Filter for analysis-related text
    keywords = ['scrambl', 'inject', 'fit', 'chi', 'sigma', 'limit', 'distribution', 
                'histogram', 'pull', 'analysis', 'sample', 'average', 'mean']
    
    print("\nRelevant text cells:")
    seen = set()
    count = 0
    for text in text_matches:
        text_lower = text.lower()
        if any(kw in text_lower for kw in keywords):
            # Clean up the text
            cleaned = text.replace('\\n', ' ').replace('\\[IndentingNewLine]', ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned not in seen and len(cleaned) > 30:
                seen.add(cleaned)
                print(f"\n  - {cleaned[:100]}...")
                count += 1
                if count >= 30:
                    break
                if count >= 30:
                    break
    
    print("\n" + "=" * 70)
    print("CODE CELLS (INPUT) MENTIONING INJECTION")
    print("=" * 70)
    
    # Simple regex for Input cells (heuristic for BoxData)
    # Cell[BoxData[ ... ], "Input"
    # This is hard to parse perfectly without a parser, but we can look for "inject" and grab context.
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'inject' in line.lower() or 'duyzinj' in line.lower():
            # Print context (5 lines before/after)
            print(f"\n--- Line {i} ---")
            context = lines[max(0, i-5):min(len(lines), i+10)]
            print('\n'.join(context))
    
    # Find Export statements
    export_pattern = r'Export\["([^"]+)"'
    exports = re.findall(export_pattern, content)
    unique_exports = list(set(exports))
    for exp in sorted(unique_exports)[:30]:
        print(f"  {exp}")


if __name__ == '__main__':
    main()
