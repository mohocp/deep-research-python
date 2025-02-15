from dataclasses import dataclass
from typing import Callable
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler, type_subscription, TopicId
from autogen_core import CancellationToken
import asyncio
import json
import math
import os
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Callable, Dict, List, Optional
import logging
from typing import Any, Callable, Dict, List, Optional
import re


from datetime import datetime


## load environment variables
load_dotenv(override=True)
LLM_KEY = os.getenv("LLM_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT") or "https://api.openai.com/v1"
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY")
FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL") or "https://api.firecrawl.com"
CONTEXT_SIZE = os.getenv("CONTEXT_SIZE") or 128000
MAX_OUTPUT_TOKENS = os.getenv("MAX_OUTPUT_TOKENS") or 8000

# logging.basicConfig(level=logging.INFO)

## initialize the model client
print(f"LLM_MODEL: {LLM_MODEL}")
print(f"LLM_ENDPOINT: {LLM_ENDPOINT}")
print(f"LLM_KEY: {LLM_KEY}")
model_client = OpenAIChatCompletionClient(
    model=LLM_MODEL,
    base_url=LLM_ENDPOINT,
    api_key=LLM_KEY,
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "max_tokens": MAX_OUTPUT_TOKENS
    },
)


## utility functions

# Set the minimum chunk size to 140
MIN_CHUNK_SIZE = 140

# Initialize the encoder using tiktoken with the "o200k_base" encoding
encoder = tiktoken.get_encoding('o200k_base')

def trim_prompt(prompt: str, context_size: int = None) -> str:
    # Use the provided context size or get it from the environment (default to 128000)
    if context_size is None:
        context_size = int(os.environ.get("CONTEXT_SIZE", 128000))
    
    if not prompt:
        return ''
    
    # Compute the number of tokens using the encoder
    token_length = len(encoder.encode(prompt))
    if token_length <= context_size:
        return prompt

    # Calculate how many tokens exceed the context size
    overflow_tokens = token_length - context_size
    # Estimate the character chunk size based on an average of 3 characters per token
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]
    
    # Use LangChain's RecursiveCharacterTextSplitter to split the prompt
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    split_chunks = splitter.split_text(prompt)
    trimmed_prompt = split_chunks[0] if split_chunks else ''
    
    # If the trimmed prompt length is equal to the original prompt,
    # perform a hard cut and try trimming again
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)
    
    # Recursively trim until the prompt is within the allowed context size
    return trim_prompt(trimmed_prompt, context_size)


def extract_json(text: str) -> str:
    """
    Extracts the JSON content from a string that may be wrapped in markdown code fences.
    If the text is wrapped in either ```json ... ``` or simply ``` ... ```, 
    it returns the content inside. Otherwise, it returns the original stripped text.
    """
    # The (?:json)? makes the literal "json" optional.
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()



def parse_json(text: str) -> dict:
    try:
        return json.loads(extract_json(text))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {text}")
        print(f"Error: {e}")
        return None

def trim_prompt(content, max_length):
    """
    Trims the content to the specified maximum length.
    If the content is shorter than max_length, it is returned unchanged.
    """
    return content[:max_length]

system_prompt = f"""You are an expert researcher. Today is {datetime.utcnow().isoformat()}. Follow these instructions when responding:
- You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
- The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
- Be highly organized.
- Suggest solutions that I didn't think about.
- Be proactive and anticipate my needs.
- Treat me as an expert in all subject matter.
- Mistakes erode my trust, so be accurate and thorough.
- Provide detailed explanations, I'm comfortable with lots of detail.
- Value good arguments over authorities, the source is irrelevant.
- Consider new technologies and contrarian ideas, not just the conventional wisdom.
- You may use high levels of speculation or prediction, just flag it for me."""


async def feedback_questions(query, num_questions=3):
    assistant = AssistantAgent("feedback", 
        model_client=model_client,
        system_message=system_prompt
        )
    user_feedback_prompt = f"""Given the following query from the user, ask some follow up questions to clarify the research direction. Return a maximum of {num_questions} questions, but feel free to return less if the original query is clear: 
                                Please return your answer as a valid JSON object with no additional text. The JSON object must exactly follow this format:
                                {{
                                    "questions": [
                                    "Your first question",
                                    "Your second question",
                                    "Your third question"
                                    ]
                                }}
                                Only output the JSON with no extra commentary.
                                <query>{query}</query>"""

    response = await assistant.on_messages([TextMessage(content=user_feedback_prompt, source="user")], CancellationToken())
    return parse_json(response.chat_message.content)["questions"]

from typing import Any, Callable, Dict, List, Optional

async def generate_serp_queries(query, learnings, num_queries=3):
    assistant = AssistantAgent("queryGenrator", 
        model_client=model_client,
        system_message=system_prompt
        )
    learnings_text = ""
    if learnings:
        learnings_text = "Here are some learnings from previous research, use them to generate more specific queries:\n" + "\n".join(learnings)

    user_serp_prompt = f"""Given the following user query, generate a list of SERP queries to research the topic. Return a maximum of {num_queries} queries (each unique). 

                            Please return your answer as a valid JSON object with no additional text. The JSON object must follow exactly this format:
                            {{
                                "queries": [
                                    {{
                                        "query": "A unique SERP query",
                                        "researchGoal": "A detailed description of what research this query is meant to accomplish"
                                    }}
                                    // up to {num_queries} items
                                ]
                            }}

                            {learnings_text}

                            <query>{query}</query>"""
    response = await assistant.on_messages([TextMessage(content=user_serp_prompt, source="user")], CancellationToken())
    return parse_json(response.chat_message.content)["queries"]


async def process_serp_result(query, results, num_learnings=3, num_follow_up_questions=3):
    assistant = AssistantAgent("processSearchResult", 
        model_client=model_client,
        system_message=system_prompt
        )
    contents = [item['markdown'] for item in results.get('data', []) if item.get('markdown')]
    # Trim each content to a maximum of 25,000 characters
    contents = [trim_prompt(content, 25000) for content in contents]

    formatted_contents = ''.join(f'<content>\n{content}\n</content>\n' for content in contents)
    prompt = f"""Given the following contents from a SERP search for the query <query>${query}</query>, generate a list of learnings from the contents. Return a maximum of ${num_learnings} learnings, but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.
    
    Please return your answer as a valid JSON object with no additional text. The JSON object must follow exactly this format:
    {{
        "learnings": [ List of learnings, max of {num_learnings}],
        "followUpQuestions": [ List of follow-up questions to research the topic further, max of ${num_follow_up_questions}]
    }}
    
    <contents>{formatted_contents}</contents>"""
    response = await assistant.on_messages([TextMessage(content=prompt, source="user")], CancellationToken())
    return parse_json(response.chat_message.content)

async def write_final_report(prompt, learnings, visited_urls):
    assistant = AssistantAgent("writeFinalReport", 
        model_client=model_client,
        system_message=system_prompt
        )
    learnings_string = trim_prompt("\n".join(f"<learning>\n{learning}\n</learning>" for learning in learnings), 150_000)

    prompt = f"""Given the following prompt from the user, write a final report on the topic using the learnings from research. 
                 Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:
                 
                

                 <prompt>{prompt}</prompt>
                 
                 Here are all the learnings from previous research:

                 <learnings>
                    {learnings_string}
                 </learnings>
                 
                 Please return your answer as a valid JSON object with no additional text. The JSON object must follow exactly this format: 
                 {{
                    "reportMarkdown": "The final report"
                 }}
                 """
    response = await assistant.on_messages([TextMessage(content=prompt, source="user")], CancellationToken())
    report_markdown = parse_json(response.chat_message.content)["reportMarkdown"]
    urls_section = "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in visited_urls)

    return report_markdown + urls_section



def log(*args):
    print(" ".join(str(arg) for arg in args))


# Set a global concurrency limit.
CONCURRENCY_LIMIT = 2

from firecrawl import FirecrawlApp
async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    learnings: Optional[List[str]] = None,
    visited_urls: Optional[List[str]] = None,
    progress: Any = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, List[str]]:
    """
    Recursively perform deep research by:
      1. Generating SERP queries based on the current query and learnings.
      2. Executing searches concurrently (limited by CONCURRENCY_LIMIT).
      3. Processing each search result to extract new learnings and URLs.
      4. If depth allows, recursively following up with new queries.
    
    Returns a dictionary with deduplicated learnings and visited URLs.
    """
    # Initialize lists if not provided.
    if learnings is None:
        learnings = []
    if visited_urls is None:
        visited_urls = []


    def report_progress(key: str, value: Any, remove: bool = False) -> None:
        if(isinstance(value, list)):
            if remove:
                progress.get(key).remove(value[0])
            progress.get(key).extend(value)
        else:
            progress[key] = value

        if on_progress:
            on_progress(progress)
    

    # Step 1: Generate SERP queries.
    report_progress('status', 'Thinking')
    serp_queries = await generate_serp_queries(query, learnings, num_queries=breadth)
    report_progress('total_queries', [serp_query['query'] for serp_query in serp_queries])
    # print(f"Generated {len(serp_queries)} queries:")
    # print(serp_queries)
    # if serp_queries:
    #     report_progress('current_query', serp_queries[0].get('query'))

    # A semaphore to limit the number of concurrent queries.
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_query(serp_query: Dict[str, Any]) -> Dict[str, List[str]]:
        async with semaphore:
            report_progress('status', 'Searching')
            report_progress('current_queries', [serp_query['query']])
            app = FirecrawlApp(api_url=FIRECRAWL_BASE_URL, api_key=FIRECRAWL_KEY)
            params = {
                'timeout': 15000,
                'limit': 5,
                "scrapeOptions": {"formats": ["markdown"]}
            }
            max_retries = 5
            retry_delay = 1  # start with a 1-second delay
            for attempt in range(max_retries):
                try:
                    # Step 2: Execute the search for this query.
                    # print(f"Running query: {serp_query['query']}")
                    result = app.search(serp_query['query'], params=params)
                    # print(f"Result for query: {serp_query['query']}")
                    # print("\n\nURL:".join(item.get('url') +"\nTitle:"+ item.get('title') for item in result.get('data', [])))
                    break  # if the call succeeds, exit the retry loop
                except Exception as e:
                    # Check for rate limiting (HTTP 429)
                    log(
                        f"Rate limit error on query: {serp_query.get('query')}. "
                        f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
            else:
                log(f"Max retries reached for query: {serp_query.get('query')}. Skipping.")
                return {'learnings': [], 'visited_urls': []}

            # Extract URLs from the search result (ignoring empty ones).
            new_urls = [
                item.get('url')
                for item in result.get('data', [])
                if item.get('url')
            ]
            report_progress('total_urls', new_urls)


            # Calculate parameters for the next recursion level.
            print(f"Breadth: {breadth}, Depth: {depth}")
            new_breadth = math.ceil(breadth / 2)
            new_depth = depth - 1

            # Step 3: Process the result to extract new learnings and follow-up questions.
            report_progress('status', 'Thinking')
            result_processed = await process_serp_result(
                query=serp_query['query'],
                results=result,
                num_follow_up_questions=new_breadth
            )
            report_progress('current_queries', [serp_query['query']], True)
            # Combine current learnings with new ones.
            all_learnings = learnings + result_processed.get('learnings', [])
            all_urls = visited_urls + new_urls

            if new_depth > 0:
                log(f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}")

                # Prepare the next query from follow-up questions.
                follow_up_questions = result_processed.get('follow_up_questions', [])
                next_query = (
                    f"Previous research goal: {serp_query.get('researchGoal')}\n"
                    f"Follow-up research directions: " + "\n".join(follow_up_questions)
                ).strip()

                report_progress('total_queries', [serp_query['query'] for serp_query in follow_up_questions])


                # Step 4: Recursively perform deep research.
                return await deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    visited_urls=all_urls,
                    progress=progress,
                    on_progress=on_progress,
                )
            else:
                return {
                    'learnings': all_learnings,
                    'visited_urls': all_urls
                }

    # Run all queries concurrently.
    tasks = [run_query(serp_query) for serp_query in serp_queries]
    results = await asyncio.gather(*tasks)

    # Combine and deduplicate the learnings and URLs from all query results.
    combined_learnings = list({learning for res in results for learning in res.get('learnings', [])})
    combined_urls = list({url for res in results for url in res.get('visited_urls', [])})

    return {
        'learnings': combined_learnings,
        'visited_urls': combined_urls
    }








