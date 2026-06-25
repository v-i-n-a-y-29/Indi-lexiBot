import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (Ensure GOOGLE_API_KEY is in your .env)
load_dotenv()

# 2. Initialize the Gemini LLM
# We use gemini-2.5-flash for fast and cost-effective generation.
# Temperature is set to 0.3 for more factual, less creative legal answers.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3, 
)

# 3. Create a simple prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You are an Indian legal research assistant focused on consumer protection, harassment-related matters, dowry-related offences, and traffic/accident law.

    Answer only from the retrieved context. Do not invent laws, citations, sections, punishments, case names, or conclusions.

    Format every answer in a clean, readable structure without markdown bold formatting.

    Required answer format:

    Short Answer:
    Give a direct 1-2 sentence answer.

    Relevant Law or Case:
    Mention the section, Act, case name, court, or judgment date if available.

    Key Points:
    - Point 1
    - Point 2
    - Point 3

    Explanation:
    Explain the legal meaning in simple language.

    Penalty / Fine / Compensation:
    If punishment, fine, limitation, or compensation is involved, show it clearly. If not applicable, skip this heading.

    Source:
    Mention the retrieved source metadata, such as Act, section, case name, file name, court, date, or page number.

    Formatting rules:
    - Do not use markdown bold.
    - Do not use double asterisks.
    - Do not wrap headings or words with **.
    - Use plain text headings only.
    - Use simple bullet points only when useful.

    If the retrieved context is insufficient, say exactly:
    "The available retrieved documents do not provide enough information to answer this confidently."

    Provide legal information, not personalized legal advice.
    """),
    ("user", "{question}")
])

# 4. Create the LangChain pipeline (Chain)
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print("Welcome to the Indian Legal Assistant! (Type 'exit' to quit)\n")
    
    while True:
        user_question = input("You: ")
        
        if user_question.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_question.strip():
            continue
            
        print("Legal Assistant is thinking...\n")
        
        print("Assistant:")
        
        # Stream the response back to the terminal
        for chunk in chain.stream({"question": user_question}):
            # StrOutputParser yields strings, so 'chunk' is just a string
            for ch in chunk:
                print(ch, end="", flush=True)
                time.sleep(0.01)
        
        print("\n" + "-" * 50 + "\n")
