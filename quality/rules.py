SCORING_RULES = {
    "readability": 30,
    "logic": 30,
    "efficiency": 20,
    "comments_bonus": 10,
    "bad_naming_penalty": -5,
    "deep_nesting_penalty": -10,
    "loop_penalty": -5
}

ERROR_EXPLANATIONS = {
    "bad_naming": {
        "title": "Poor Variable Naming",
        "explanation": "Single-letter variable names reduce code clarity. Use descriptive names like 'count', 'user_name', or 'temperature' instead of 'i', 'x', or 'n' to make your code self-documenting and easier to maintain.",
        "example": "x = 10; instead user_age = 10"
    },
    "no_comments": {
        "title": "Missing Code Comments",
        "explanation": "Comments help others (and future you!) understand the code's purpose. Add comments for complex logic, important decisions, and non-obvious implementations.",
        "example": "No explanation; instead # Calculate total after applying discount"
    },
    "too_many_loops": {
        "title": "Excessive Loop Usage",
        "explanation": "Using more than 3 loops suggests over-complication. Consider using built-in functions, list comprehensions, or algorithmic optimization to simplify your logic.",
        "example": "5 nested loops; instead Use filter(), map(), or numpy operations"
    },
    "infinite_loop": {
        "title": "Potential Infinite Loop",
        "explanation": "'while True' without proper exit conditions can freeze your program. Ensure you have break statements or condition changes to exit the loop.",
        "example": " while True: ...; instead while condition: ... break"
    },
    "deep_nesting": {
        "title": "Deep Code Nesting",
        "explanation": "Deeply nested code (multiple for/if inside each other) makes logic hard to follow. Refactor into separate functions or use flatmap operations.",
        "example": " O(n²) nested loops; instead Single-pass algorithm"
    }
}

HINT_TEMPLATES = {
    "bad_naming": [
        "Step 1: Identify single-letter or unclear variable names in your code",
        "Step 2: Think about what each variable represents (count? index? value?)",
        "Step 3: Rename using descriptive, meaningful names (use snake_case in Python)"
    ],
    "no_comments": [
        "Step 1: Find complex or non-obvious code sections",
        "Step 2: Think about WHY this logic is needed, not just WHAT it does",
        "Step 3: Add comments explaining the purpose and any important details"
    ],
    "too_many_loops": [
        "Step 1: Count the number of loops in your code",
        "Step 2: Check if you can combine loops or use built-in functions",
        "Step 3: Consider using map(), filter(), list comprehensions, or vectorized operations"
    ],
    "infinite_loop": [
        "Step 1: Find 'while True' statements in your code",
        "Step 2: Identify what condition should end this loop",
        "Step 3: Add a break statement or change the condition to exit safely"
    ],
    "deep_nesting": [
        "Step 1: Look at your nested loops - how many levels deep are they?",
        "Step 2: Consider if you can extract inner logic into a separate function",
        "Step 3: Use a more efficient algorithm (e.g., hash tables instead of nested loops)"
    ]
}