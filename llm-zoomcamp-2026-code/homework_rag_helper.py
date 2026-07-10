from ingest import load_faq_data, build_index
import tiktoken

documents = load_faq_data()
index = build_index(documents)

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question : {question}
Context : {context}
"""

from minsearch import Index

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model  


    def search(self, query, num_results=5):
        boost_dict = {"content": 3.0, "filename": 0.5}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append("File: " + doc["filename"])
            lines.append("content: " + doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )


    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def get_token_count(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # If the model is not found, default to cl100k_base
            # which is the standard for GPT-4 and GPT-3.5 models
            encoding = tiktoken.get_encoding("cl100k_base")
            
        return len(encoding.encode(text))

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        # Calculate tokens before sending
        token_count = self.get_token_count(prompt)
        print(f"--- Sending {token_count} input tokens to the model ---")

        answer = self.llm(prompt)
        return answer
    

