from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    
    prompt = PromptTemplate(
        template = """
        You are a helpful and knowledgeable medical assistant.
        
        Answer ONLY from the provided context.
        If the context is insufficient, respond with: "I don't know."
        
        When answering, always follow this structure:
        
        1. **Why it is happening:** Explain the underlying reason or cause clearly. (MUST)
        2. **Prevention:** Describe what the person can do to prevent or reduce it. (MUST)
        3. **Home Remedies:** List safe and effective home remedies. (MUST)
        4. **Medicines/Supplements:** Mention only if context provides them. Include:
           - Whether it is harmful or not.
           - Timing and usage instructions (e.g., morning/night, before/after food, empty stomach). (IF ANY)
        5. **Physical Exercises:** Suggest only if relevant and provided in context. (IF ANY)
        6. **Dietary Plan:** Include foods, fruits, or dry fruits to add or avoid. (IF ANY)
        
        Context:
        {context}
        
        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )
    
    parser = StrOutputParser()
    
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
