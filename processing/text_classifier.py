from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
non_edu_keywords = ["movie", "netflix", "watch", "download", "shopping", "travel","celebrity"]
edu_keywords = ["explain", "why", "how", "science", "language", "history", "logic"]

def is_educational(question):
    if any(keyword in question.lower() for keyword in non_edu_keywords):
        return False
    
    if any(keyword in question.lower() for keyword in edu_keywords):
        return True
    
    labels = [
        "academic, general knowledge, critical thinking, language, or science",
        "movies, shopping, travel, or personal lifestyle"
    ]
    result = classifier(question, labels)
    
    return result["scores"][0] > 0.5

# Test
questions =[
   "What is the Pythagorean theorem?",  
    "Where to watch free movies?",     
    "How to write an essay?"   
    ]
for q in questions:
    print(f"Q: {q}")
    print("Educational?" if is_educational(q) else "Non-educational")
    print("---")