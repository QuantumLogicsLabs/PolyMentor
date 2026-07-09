from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_pretrained()

result = mentor.analyze(
    """
for i in range(10):
    print(i)
    if i = 5:
        break
""",
    language="python",
    level="beginner",
)

print("Error types:    ", result.error_types)
print("Primary error:  ", result.primary_error)
print("Explanation:    ", result.explanation)
print("Hint:           ", result.hints[0])
print("Concept taught: ", result.concept_taught)
print("Quality score:  ", result.quality_score)
