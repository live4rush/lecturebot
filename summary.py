from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_summary(llm, input_text):
    # Chunk the input_text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(input_text.replace('\n', ''))
    print("Number of chunks:",len(chunks))

    # Map
    map_template = """[INST] <<SYS>>
        You are a helpful assistant. Complete the task below. Output the answer only. Do not include any greetings or instructions.
        <</SYS>>
        Summarize the below text from a class trascript:
        {docs}[/INST]
        """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Process each chunk through the map_chain
    summaries = []
    for i,chunk in enumerate(chunks):
        print("this is chunk:",i)
        summary = map_chain.run(chunk)
        cleaned_summary = summary.replace(
            "Sure! Here is a summary of the text:\n\n", "")
        summaries.append(cleaned_summary)

    # Concatenate summaries into one string
    concatenated_summaries = ' '.join(summaries)
    output = concatenated_summaries.replace(
        "Sure! Here is the summary of the class trascript:\n\n", "")
    output = concatenated_summaries.replace(
        "Sure! Here is a summary of the class trascript:\n\n", "")

    return output
