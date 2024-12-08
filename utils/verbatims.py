import os
import logging
import yaml
import unicodedata
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm_utils import LLMUtils


class Sentiment(str, Enum):
    NEGATIVE = "Négatif"
    NEUTRAL = "Neutre"
    POSITIVE = "Positif"

# Verbatim classifier class
class VerbatimClassifier:
    """
    A class to interact with the Azure OpenAI model for text classification tasks.
    """

    def __init__(self, llm):
        """
        Initializes the VerbatimClassifier with an LLM instance.
        
        Args:
            llm (AzureChatOpenAI): The LLM instance to use.
        """
        self.llm = llm
        self.parser_classifier = None
        self.prompt_llm = None
        self.prompt_format = None
        self.llm_classify = None
        self.set_output = None
        self.themes = []
        self.theme_enum = None
        self._load_data()
        self._initialize_parser()
        self._load_prompts()

    def _load_data(self):
        """
        Loads the themes and descriptions from YAML configuration files.
        """
        with open('config/themes.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        self.themes = data['themes']

        list_themes = {self._normalize_text(theme['name']): theme['name'] for theme in self.themes}
        self.theme_enum = Enum("Theme", list_themes)

    def _load_prompts(self):
        """
        Loads the prompt templates from YAML configuration files.
        """
        with open('config/prompts.yaml', 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)

        self.prompt_llm = PromptTemplate(
            template=prompts['prompts']['classify']['text'],
            input_variables=['comment', 'themes', 'descriptions'],
            partial_variables={"format_instructions": self.parser_classifier.get_format_instructions()},
        )
        self.prompt_format = PromptTemplate(
            template=prompts['prompts']['correct_output']['text'],
            input_variables=['input'],
            partial_variables={"format_instructions": self.parser_classifier.get_format_instructions()},
        )
        self.llm_classify = LLMChain(prompt=self.prompt_llm, llm=self.llm, output_parser=self.parser_classifier)
        self.set_output = LLMChain(prompt=self.prompt_format, llm=self.llm, output_parser=self.parser_classifier)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalizes a text string by removing special characters and spaces.
        
        Args:
            text (str): The text to normalize.
        Returns:
            str: The normalized text.
        """
        return u"".join([c for c in unicodedata.normalize('NFKD', text)
                         .replace(" ", "_")
                         .replace('/', '')
                         .replace('(', '')
                         .replace(')', '') if not unicodedata.combining(c)])

    def _initialize_parser(self):
        """
        Initializes the Pydantic output parser with the correct Verbatim model.
        """
        theme_pydantics = self.theme_enum

        class Verbatim(BaseModel):
            verbatim: str
            theme: theme_pydantics
            sentiment: Sentiment

        class OutputLLMCustom(BaseModel):
            results: List[Verbatim]

        self.parser_classifier = PydanticOutputParser(pydantic_object=OutputLLMCustom)
        self.Verbatim = Verbatim

    def classify_comment(self, comment: str) -> List[BaseModel]:
        """
        Classifies a given comment into predefined themes and sentiments.
        
        Args:
            comment (str): The comment to classify.
        
        Returns:
            List[BaseModel]: The classification results.
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
                raise Exception("LLM functionnal error")

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
        logging.info(f"Error: {e}")