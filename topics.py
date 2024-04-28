from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re


def check_summary_token_size(llm, summary):
    # Tokenize the summary
    tokens = llm.get_num_tokens(summary)

    # Keep truncating until token count is less than 3800
    while tokens > 3800:
        excess_length = tokens - 3800
        # Calculate the number of characters to remove from the beginning and end
        chars_to_remove = excess_length // 2
        summary = summary[chars_to_remove:-chars_to_remove]
        # Re-tokenize the truncated summary
        tokens = llm.get_num_tokens(summary)

    return summary


def extract_topics(text):
    # Regular expression to match the pattern "1. subject 2. subject 3. subject, etc."
    pattern = r'\d+\.\s(.*?)(?=\n\d+\.|$)'

    # Find all matches using the pattern
    topics = re.findall(pattern, text, re.DOTALL | re.MULTILINE)

    # Strip any trailing whitespace and return the list of subjects
    return [topic.strip() for topic in topics]


def generate_topics(llm, summary):
    # Map
    map_template = """[INST] <<SYS>>
        You are a helpful assistant. Complete the task below. Output the result in a numbered list. Eaxmple: 1. Topic 1 2. Topic 2
        <</SYS>>
        Make a list three topics the class transcript covers based on the summary below.
        {summary}[/INST]
        """
    map_prompt = PromptTemplate.from_template(template=map_template)

    # llm.grammar = "list.gbnf"
    # llm.grammar_path = "./grammars"
    map_chain = LLMChain(llm=llm,
                         prompt=map_prompt
                         )

    summary = check_summary_token_size(llm, summary)

    topics = map_chain.run(summary)

    return extract_topics(topics)
