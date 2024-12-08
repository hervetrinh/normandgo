from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

import time

llm = AzureChatOpenAI(
            model=os.getenv("AZURE_NAME"),
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_PUBLIC_ENDPOINT"),
            openai_api_version=os.getenv('AZURE_API_VERSION'),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
            temperature=0.7
        )


if __name__ == '__main__':
    start = time.time()
    print(llm.invoke("1+1?").content)
    print(time.time() - start)