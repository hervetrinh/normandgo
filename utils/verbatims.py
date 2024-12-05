import os
import logging
import unicodedata
from enum import Enum
from typing import List
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm_utils import LLMUtils
from config import config


class Sentiment(str, Enum):
    NEGATIVE = "Négatif"
    NEUTRAL = "Neutre"
    POSITIVE = "Positif"


class VerbatimClassifier:
    """
    A class to interact with the Azure OpenAI model for text classification tasks.
    """

    def __init__(self, llm):
        """
        Initializes the VerbatimClassifier with an LLM instance.
        """
        self.llm = llm
        self.parser_classifier = None
        self.prompt_llm = None
        self.prompt_format = None
        self.llm_classify = None
        self.set_output = None
        self.themes = config['themes']
        self.theme_enum = None
        self._initialize_parser()
        self._load_prompts()

    def _load_prompts(self):
        """
        Loads the prompt templates from the Python configuration.
        """
        prompts = config['prompts']
        self.prompt_llm = PromptTemplate(
            template=prompts['classify']['text'],
            input_variables=['comment', 'themes', 'descriptions'],
            partial_variables={"format_instructions": self.parser_classifier.get_format_instructions()},
        )
        self.prompt_format = PromptTemplate(
            template=prompts['correct_output']['text'],
            input_variables=['input'],
            partial_variables={"format_instructions": self.parser_classifier.get_format_instructions()},
        )
        self.llm_classify = LLMChain(prompt=self.prompt_llm, llm=self.llm, output_parser=self.parser_classifier)
        self.set_output = LLMChain(prompt=self.prompt_format, llm=self.llm, output_parser=self.parser_classifier)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalizes a text string by removing special characters and spaces.
        """
        return u"".join(
            [c for c in unicodedata.normalize('NFKD', text)
             .replace(" ", "_")
             .replace('/', '')
             .replace('(', '')
             .replace(')', '') if not unicodedata.combining(c)]
        )

    def _initialize_parser(self):
        """
        Initializes the Pydantic output parser with the correct Verbatim model.
        """
        list_themes = {self._normalize_text(theme['name']): theme['name'] for theme in self.themes}
        self.theme_enum = Enum("Theme", list_themes)

        class Verbatim(BaseModel):
            verbatim: str
            theme: self.theme_enum
            sentiment: Sentiment

        class OutputLLMCustom(BaseModel):
            results: List[Verbatim]

        self.parser_classifier = PydanticOutputParser(pydantic_object=OutputLLMCustom)

    def classify_comment(self, comment: str) -> List[BaseModel]:
        """
        Classifies a given comment into predefined themes and sentiments.
        """
        themes = [theme['name'] for theme in self.themes]
        descriptions = [theme['description'] for theme in self.themes]

        try:
            result_llm = self.llm_classify.invoke({
                'comment': comment,
                'themes': themes,
                'descriptions': '- ' + '- '.join(descriptions)
            })
            return result_llm['text'].dict()['results']
        except Exception as e1:
            try:
                result_llm = self.set_output.invoke({'input': e1})
                return result_llm['text'].dict()['results']
            except Exception as e2:
                raise Exception("LLM functional error")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Initialization...")
    llm_utils = LLMUtils()
    classifier = VerbatimClassifier(llm=llm_utils.get_llm())
    comment = "Très beau patrimoine ! Mais il fait très froid à l'intérieur quand même..."
    
    logging.info("Launch classifier...")
    try:
        result = classifier.classify_comment(comment)
        logging.info(result)
    except Exception as e:
        logging.error(f"Error: {e}")
