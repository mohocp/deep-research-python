import asyncio
from typing import Dict, Any
from deep_research import feedback_questions, deep_research, write_final_report
import os
import dotenv

dotenv.load_dotenv()

BREADTH = 4 #os.getenv("BREADTH") or 4
DEPTH = 2 #os.getenv("DEPTH") or 2


def print_list(list):
    for index, item in enumerate(list):
        print(f"{index+1}: {item}")

def log_progress(progress: Dict[str, Any]):
    print("========================================================")
    print(f"Status: {progress.get('status')}")
    print(f"Total URLs: {len(progress.get('total_urls', []))}")
    print_list(progress.get('total_urls', []))
    print(f"Total Queries: {len(progress.get('total_queries', []))}")
    print_list(progress.get('total_queries', []))
    print(f"Current Queries: {len(progress.get('current_queries', []))}")
    print_list(progress.get('current_queries', []))
    print("========================================================")

async def main():

        # A simple progress state that can be reported to a callback.
    progress: Dict[str, Any] = {
        'total_urls': [],
        'total_queries': [],
        'current_queries': [],
        'status': 'initializing'
    }

    # query = "Search for best tools to do GenAI-native document parser that can parse complex document data for any downstream LLM use case and ensure the parser is open source"
    query = input("Enter your query: ")
    questions = await feedback_questions(query)
    answers = []
    for question in questions:
        answer = input(f"{question}:\n")
        answers.append(answer)

    joined_qna = "\n".join([f"- Question: {question}\n- Answer: {answer}" for question, answer in zip(questions, answers)])

    combined_query = f"""User query: {query}
                        Follow up questions and answers:
                        {joined_qna}"""
    result = await deep_research(combined_query, breadth=BREADTH, depth=DEPTH, progress=progress, on_progress=log_progress)
    report = await write_final_report(query, result["learnings"], result["visited_urls"])
    with open('output.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n\nFinal Report:\n\n{report}")
    print("\nReport has been saved to output.md")

if __name__ == "__main__":
    asyncio.run(main())
