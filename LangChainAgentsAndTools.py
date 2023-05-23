import credentials
from langchain import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.tools import Tool,BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

#Initializing Open AI LLM
llm = OpenAI(
    openai_api_key=credentials.openAIAPIKey,
    temperature=0
)

#LLM Math Tool
tools = load_tools(
    ["llm-math"], 
    llm=llm
)

#Integrating Google Custom Search API Tool
os.environ["GOOGLE_CSE_ID"] = credentials.GoogleCSE_ID
os.environ["GOOGLE_API_KEY"] = credentials.GoogleSearchAPI

search = GoogleSearchAPIWrapper()

searchTool = Tool(
    name = "Google Search",
    description=''' Use this tool when you need to answer questions regarding current affairs''',
    func=search.run
)

tools.append(searchTool)

#Adding Custom tool to generate Caption for images

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = ''' use this tool when given the URL of an image that you'd like to generate
                      a caption for'''

    def _run(self, url: str):
        #define the model to use
        hf_model = "Salesforce/blip-image-captioning-large"
        #set GPU run time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = BlipProcessor.from_pretrained(hf_model)
        model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
        # download the image and convert to PIL object
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
        # generate the caption
        out = model.generate(**inputs, max_new_tokens=20)
        # get the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


tools.append(ImageCaptionTool())

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description", 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
)

#To view the promt
print(zero_shot_agent.agent.llm_chain.prompt.template)

query = "Add your Query"

zero_shot_agent(query)
