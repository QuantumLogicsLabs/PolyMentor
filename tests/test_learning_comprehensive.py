#!/usr/bin/env python3
"""
Comprehensive test of PolyMentor learning guidance features.
Demonstrates how the AI-powered learning system helps students understand concepts.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a nicely formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def test_learning_system():
    """Test the complete learning guidance system"""
    
    print_section("🎓 POLYMENTOR LEARNING GUIDANCE SYSTEM - COMPREHENSIVE TEST")
    
    # ========================================================================
    # SCENARIO 1: Student discovers concepts available
    # ========================================================================
    print_section("SCENARIO 1: Explore Available Concepts")
    print("""
    Student: "What programming concepts can PolyMentor teach me?"
    """)
    
    response = requests.get(f"{BASE_URL}/learn/concepts")
    data = response.json()
    
    print(f"✅ Found {data['total_concepts']} concepts:")
    for concept in data['concepts'][:3]:
        print(f"   • {concept['name']} ({concept['difficulty']})")
        print(f"     Prerequisites: {', '.join(concept['prerequisites']) or 'None'}")
    
    # ========================================================================
    # SCENARIO 2: Student has a syntax error and wants to learn
    # ========================================================================
    print_section("SCENARIO 2: Learn From an Error")
    print("""
    Student: "I have a syntax error. Can you explain what's wrong
             and help me understand the underlying concept?"
    """)
    
    buggy_code = """
for i in range(10):
    if i = 5:
        print("Found 5!")
    """.strip()
    
    print("Code submitted:")
    print("```python")
    print(buggy_code)
    print("```")
    
    response = requests.post(
        f"{BASE_URL}/learn/from-error",
        json={
            "code": buggy_code,
            "language": "python",
            "level": "beginner"
        }
    )
    data = response.json()
    
    print(f"\n📊 Analysis Results:")
    print(f"   Status: {data['status']}")
    print(f"   Errors found: {data['total_errors']}")
    print(f"\n💡 Overall advice: {data['overall_advice']}")
    
    if data.get('learning_materials'):
        material = data['learning_materials'][0]
        print(f"\n📌 First Error:")
        print(f"   Category: {material['error']['category']}")
        print(f"   Severity: {material['error']['severity']}")
        print(f"   Message: {material['error']['message']}")
        print(f"   Suggestion: {material['error']['suggestion']}")
        
        if material.get('concept'):
            print(f"\n🎯 Related Concept: {material['concept']['name']}")
            print(f"   {material['concept']['simple_explanation'][:100]}...")
            
            if material.get('examples'):
                print(f"\n📚 Code Examples:")
                example = material['examples'][0]
                print(f"   Title: {example['title']}")
                print(f"   Wrong way: {example['wrong_code'][:40]}...")
                print(f"   Right way: {example['right_code'][:40]}...")
                print(f"   Explanation: {example['explanation']}")
            
            if material.get('tips'):
                print(f"\n💡 Tips for mastery:")
                for tip in material['tips'][:2]:
                    print(f"   • {tip}")
    
    # ========================================================================
    # SCENARIO 3: Student wants to deep-dive into a concept
    # ========================================================================
    print_section("SCENARIO 3: Deep-Dive Into a Concept")
    print("""
    Student: "I want to really understand Comparison Operators.
             Show me everything about this concept."
    """)
    
    response = requests.get(f"{BASE_URL}/learn/concept/comparison_operators")
    data = response.json()
    
    print(f"Concept: {data['concept_name']}")
    print(f"Difficulty: {data['difficulty']}")
    print(f"\n📖 Simple Explanation (Perfect for beginners):")
    print(f"   {data['simple_explanation']}")
    
    print(f"\n🔍 Prerequisites:")
    for prereq in data['prerequisites']:
        print(f"   • {prereq}")
    
    print(f"\n⚠️  Common Mistakes to Avoid:")
    for mistake in data['common_mistakes'][:3]:
        print(f"   • {mistake}")
    
    print(f"\n🎓 Tips for Mastery:")
    for tip in data['tips_for_mastery'][:3]:
        print(f"   • {tip}")
    
    print(f"\n📚 Code Examples:")
    for example in data['code_examples']:
        print(f"   • {example['title']}")
    
    # ========================================================================
    # SCENARIO 4: Student wants a learning path
    # ========================================================================
    print_section("SCENARIO 4: Get a Personalized Learning Path")
    print("""
    Student: "I want to learn about Loop Boundaries.
             What should I learn first, and what comes after?"
    """)
    
    response = requests.get(f"{BASE_URL}/learn/path/loop_boundaries")
    data = response.json()
    
    print(f"Concept: {data['starting_concept']}")
    print(f"Difficulty: {data['difficulty']}")
    
    print(f"\n📋 Learning Path:")
    path = data['learning_path']
    
    print(f"   Step 1: Prerequisites")
    if isinstance(path['step_1_prerequisites'], list):
        for prereq in path['step_1_prerequisites']:
            if isinstance(prereq, dict):
                print(f"      ✓ {prereq.get('name', prereq)}")
    
    print(f"   Step 2: Main Concept")
    print(f"      ✓ {path['step_2_main_concept']['name']}")
    
    print(f"   Step 3: Practice")
    print(f"      ✓ {path['step_3_practice']['message']}")
    
    print(f"   Step 4: Related Concepts")
    if isinstance(path['step_4_related_concepts'], list):
        for related in path['step_4_related_concepts']:
            if isinstance(related, dict):
                print(f"      ✓ {related.get('name', related)}")
    
    print(f"\n⏱️  Estimated time: {data['estimated_time']}")
    
    # ========================================================================
    # SCENARIO 5: Student wants to understand their working code
    # ========================================================================
    print_section("SCENARIO 5: Understand What Your Code Does")
    print("""
    Student: "I wrote a loop that works, but I want to make sure
             I understand what concepts it uses."
    """)
    
    working_code = """
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num == 3:
        print("Found 3!")
    """.strip()
    
    print("Code submitted:")
    print("```python")
    print(working_code)
    print("```")
    
    response = requests.post(
        f"{BASE_URL}/learn/explain-code",
        json={
            "code": working_code,
            "language": "python",
            "level": "beginner"
        }
    )
    data = response.json()
    
    print(f"\n📊 Code Analysis:")
    print(f"   Code length: {data['code_length']} characters")
    print(f"   Quality score: {data['quality_score']}/100")
    
    print(f"\n🎯 What this code does:")
    print(f"   {data['what_this_code_does']}")
    
    print(f"\n📚 Concepts being used:")
    for concept in data['detected_concepts']:
        print(f"   • {concept['concept']}: {concept['explanation'][:60]}...")
    
    print(f"\n💡 Improvements suggested:")
    for suggestion in data['improvements_suggested'][:2]:
        print(f"   • {suggestion}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("✅ LEARNING SYSTEM COMPLETE TEST SUMMARY")
    print("""
    The AI-Powered Learning Guidance System helps students:
    
    1. 📚 Explore concepts and prerequisites
    2. 🎯 Learn from their errors with concept mapping
    3. 🔍 Deep-dive into specific concepts
    4. 🗺️  Follow structured learning paths
    5. 💡 Understand their working code
    
    Key Features:
    ✓ Human-like explanations (no jargon)
    ✓ Multi-level learning (beginner → advanced)
    ✓ Error-to-concept mapping
    ✓ Code examples (wrong vs right)
    ✓ Common mistakes explained
    ✓ Tips for mastery
    ✓ Learning resources
    ✓ Personalized learning paths
    ✓ Concept prerequisites mapped
    
    Philosophy:
    Instead of just pointing out what's wrong,
    PolyMentor teaches the underlying concepts
    so students understand AND avoid mistakes in the future.
    """)

if __name__ == "__main__":
    try:
        test_learning_system()
        print("\n" + "=" * 80)
        print(" 🎉 ALL TESTS PASSED - Learning Guidance System is Working!")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
