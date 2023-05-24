import os
from langchain import PromptTemplate, OpenAI
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from Levenshtein import distance


# TODO: Add a preprocessor to select better the context: resume, personal data, or cover letter.
class GPTAnswerer:
    def __init__(self, resume: str, personal_data: str, cover_letter: str = "", job_description: str = ""):
        self.resume = resume
        self.personal_data = personal_data
        self.cover_letter = cover_letter
        self.job_description = job_description
        self.llm = OpenAI(model_name="text-davinci-003", openai_api_key=GPTAnswerer.openai_api_key(), temperature=0.5, max_tokens=-1)

        # TODO: Summarize the job description.

    @staticmethod
    def openai_api_key():
        """
        Returns the OpenAI API key.
        environment variable used: OPEN_AI_API_KEY
        Returns: The OpenAI API key.
        """
        key = os.getenv('OPEN_AI_API_KEY')

        if key is None:
            raise Exception("OpenAI API key not found. Please set the OPEN_AOI_API_KEY environment variable.")

        return key

    def summarize_job_description(self, text: str) -> str:
        """
        Summarizes the text using the OpenAI API.
        Args:
            text: The text to summarize.
        Returns:
            The summarized text.
        """
        summarize_prompt_template = """
        Summarize the following job description, removing boilerplate text:
        
        ```
        {text}
        ```
        
        ---
        
        Summary:"""
        prompt = PromptTemplate(input_variables=["text"], template=summarize_prompt_template)       # Define the prompt (template)
        formatted_prompt = prompt.format_prompt(text=text)                                          # Format the prompt with the data
        output = self.llm(formatted_prompt.to_string())                                             # Send the prompt to the llm

        return output

    def answer_question_textual_wide_range(self, question: str) -> str:
        # Can answer questions from the resume, personal data, and cover letter. Deciding which context is relevant. So we don't create a very large prompt concatenating all the data.
        # Prompt templates:
        # - Resume stuff + Personal data.
        # - Cover letter -> personalize to the job description
        # - Summary -> Resume stuff + Job description (summary)

        # Templates:
        # - Resume Stuff
        resume_stuff_template = """
        The following is a resume, personal data, and an answered question using this information, being answered by the person who's resume it is (first person).
        
        ## Rules
        - Answer questions directly (if possible). eg. "Full Name" -> "John Oliver".
        - If seems likely that you have the experience, even if is not explicitly defined, answer as if you have the experience.
        - If you cannot answer the question, answer things like "I have no experience with that, but I learn fast, very fast", "not yet, but I will learn"...
        - The answer must not be longer than a tweet (140 characters).
        
        ## Example
        Resume: I'm a software engineer with 10 years of experience on both swift and python.
        Question: What is your experience with swift?
        Answer: 10 years.
        
        -----
        
        ## Extended personal data:
        ```
        {personal_data}
        ```
        
        ## Resume:
        ```
        {resume}
        ```
                
        ## Question:
        {question}
        
        ## Answer:"""

        # - Cover Letter
        cover_letter_template = """
        The following is a cover letter, a job description, and an answered question using this information, being answered by the person who's signing the cover letter (first person).
        
        ## Rules
        - If the question is "cover letter" answer the cover letter, 
        - Slightly modify the cover letter to personalize it to the job description if needed.
        - The answer must not be longer than two tweets (280 characters).
        
        ## Job Description:
        ```
        {job_description}
        ```
        
        ## Cover Letter:
        ```
        {cover_letter}
        ```

        ## Question:
        {question}

        ## Answer:"""

        # - Summary
        summary_template = """
        The following is a resume, a job description, and an answered question using this information, being answered by the person who's resume it is (first person).
        
        ## Rules
        - Answer questions directly.
        - If seems likely that you have the experience, even if is not explicitly defined, answer as if you have the experience.
        - Find relations between the job description and the resume, and answer questions about that.
        
        ## Job Description:
        ```
        {job_description}
        ```
        
        ## Resume:
        ```
        {resume}
        ```
        
        ## Question:
        {question}
        
        ## Answer:"""

        prompt_infos = [
            {
                "name": "resume",
                "description": "Good for answering questions about job experience, skills, education, and personal data. Questions like 'experience with python', 'education', 'full name', etc.",
                "prompt_template": resume_stuff_template
            },
            {
                "name": "cover letter",
                "description": "Addressing questions about the cover letter and personal characteristics about the role. Questions like 'cover letter', 'why do you want to work here?', etc.",
                "prompt_template": cover_letter_template
            },
            {
                "name": "summary",
                "description": "Good for answering questions about the job description, and how I will fit into the company. Questions like, summary of the resume, why I'm a good fit, etc.",
                "prompt_template": summary_template
            }
        ]

        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["personal_data", "resume", "cover_letter", "job_description", "question"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=self.llm, output_key="text")

        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["personal_data", "resume", "cover_letter", "job_description", "question"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)

        return chain.run(question)

    def answer_question_textual(self, question: str) -> str:
        template = """The following is a resume and an answered question about the resume, being answered by the person who's resume it is (first person).
        
        ## Rules
        - Answer the question directly, if possible.
        - If seems likely that you have the experience based on the resume, even if is not explicit on the resume, answer as if you have the experience.
        - If you cannot answer the question, answer things like "I have no experience with that, but I learn fast, very fast", "not yet, but I will learn".
        - The answer must not be larger than a tweet (140 characters).
        - Answer questions directly. eg. "Full Name" -> "John Oliver".

        ## Example
        Resume: I'm a software engineer with 10 years of experience on both swift and python.
        Question: What is your experience with swift?
        Answer: 10 years.
        
        -----
        
        ## Extended personal data:
        ```
        {personal_data}
        ```
        
        ## Resume:
        ```
        {resume}
        ```
                
        ## Question:
        {question}
        
        ## Answer:"""

        prompt = PromptTemplate(input_variables=["personal_data", "resume", "question"], template=template)                 # Define the prompt (template)
        formatted_prompt = prompt.format_prompt(personal_data=self.personal_data, resume=self.resume, question=question)    # Format the prompt with the data
        output = self.llm(formatted_prompt.to_string())                                                                     # Send the prompt to the llm

        return output

    def answer_question_numeric(self, question: str, default_experience: int = 0) -> int:
        template = """The following is a resume and an answered question about the resume, the answer is an integer number.
        
        ## Rules
        - The answer must be an integer number.
        - The answer must only contain digits.
        - If you cannot answer the question, answer {default_experience}.
        
        ## Example
        Resume: I'm a software engineer with 10 years of experience on swift and python.
        Question: How many years of experience do you have on swift?
        Answer: 10
        
        -----
        
        ## Extended personal data:
        ```
        {personal_data}
        ```
        
        ## Resume:
        ```
        {resume}
        ```

        ## Question:
        {question}
        
        ## Answer:"""

        prompt = PromptTemplate(input_variables=["default_experience", "personal_data", "resume", "question"], template=template)                                   # Define the prompt (template)
        formatted_prompt = prompt.format_prompt(personal_data=self.personal_data, resume=self.resume, question=question, default_experience=default_experience)     # Format the prompt with the data
        output_str = self.llm(formatted_prompt.to_string())                     # Send the prompt to the llm
        # Convert to int with error handling
        try:
            output = int(output_str)                                            # Convert the output to an integer
        except ValueError:
            output = default_experience                                         # If the output is not an integer, return the default experience
            # Print error message
            print(f"Error: The output of the LLM is not an integer number. The default experience ({default_experience}) will be returned instead. The output was: {output_str}")

        return output

    def answer_question_from_options(self, question: str, options: list[str]) -> str:
        template = """The following is a resume and an answered question about the resume, the answer is one of the options.
        
        ## Rules
        - The answer must be one of the options.
        - The answer must exclusively contain one of the options.
        - Answer the option that seems most likely based on the resume.
        
        ## Example
        Resume: I'm a software engineer with 10 years of experience on swift, python, C, C++.
        Question: How many years of experience do you have on python?
        Options: [1-2, 3-5, 6-10, 10+]
        Answer: 10+
        
        -----
        
        ## Extended personal data:
        ```
        {personal_data}
        ```

        ## Resume:
        ```
        {resume}
        ```

        ## Question:
        {question}
        
        ## Options:
        {options}
        
        ## Answer:"""

        prompt = PromptTemplate(input_variables=["personal_data" "resume", "question", "options"], template=template)                # Define the prompt (template)
        formatted_prompt = prompt.format_prompt(personal_data=self.personal_data, resume=self.resume, question=question, options=options)   # Format the prompt with the data

        output = self.llm(formatted_prompt.to_string())                 # Send the prompt to the llm

        # Guard the output is one of the options
        if output not in options:
            # Choose the closest option to the output, using a levenshtein distance
            closest_option = min(options, key=lambda option: distance(output, option))
            output = closest_option
            print(f"Error: The output of the LLM is not one of the options. The closest option ({closest_option}) will be returned instead. The output was: {output}, options were: {options}")

        return output
