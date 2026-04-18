from scorer import evaluate_code, ExplanationGenerator, HintSystem

# Sample code with issues
sample_code = """
for i in range(10):
    for j in range(10):
        print(i, j)
while True:
    x = x + 1
"""


print("\nPOLYMENTOR CODE QUALITY EVALUATION\n")

# 1. EVALUATE CODE - Quality Scoring
print("\n📊 CODE EVALUATION RESULTS:")

result = evaluate_code(sample_code)

print(f"Total Score: {result['total_score']}/80")
print(f"\nScore Breakdown:")
print(f"  • Readability: {result['breakdown']['readability']}/30")
print(f"  • Logic: {result['breakdown']['logic']}/30")
print(f"  • Efficiency: {result['breakdown']['efficiency']}/20")

# 2. EXPLANATIONS - Clear Error Explanations
print("\n\nERROR EXPLANATIONS:")

if result['detailed_issues']:
    for i, issue in enumerate(result['detailed_issues'], 1):
        print(f"\n{i}. {issue['explanation']['title']} [Severity: {issue['severity'].upper()}]")
        print(f"   Score Impact: {issue['score_impact']:+d}")
        print(f"   Explanation: {issue['explanation']['explanation']}")
        print(f"   Example: {issue['explanation']['example']}")
else:
    print("✓ No issues detected!")

# 3. HINTS - Step-by-Step Guidance
print("\n\nLEARNING HINTS:")

if result['detailed_issues']:
    for issue in result['detailed_issues']:
        print(f"\n{issue['explanation']['title']}:")
        for hint in issue['hints']:
            print(f"   {hint}")
else:
    print("✓ Your code looks great! No hints needed.")

# 4. Summary Feedback
print(f"\n\n{result['summary']}")


# Demonstrate individual features
print("\n\nFEATURE DEMONSTRATION:")


print("\n1️⃣  EXPLANATION GENERATOR:")
explanation = ExplanationGenerator.get_explanation("bad_naming")
print(f"   Error Type: {explanation['title']}")
print(f"   Details: {explanation['explanation'][:60]}...")

print("\n2️⃣  HINT SYSTEM:")
hints = HintSystem.get_hints("too_many_loops")
print(f"   Topic: Reducing Loop Count")
for i, hint in enumerate(hints, 1):
    print(f"   {hint}")

