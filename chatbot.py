from dotenv import load_dotenv

load_dotenv()

import nest_asyncio
nest_asyncio.apply()

from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from os import path

years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set={}
all_docs=[]

# for year in years:
#     year_docs = loader.load_data(
#         file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
#     )
#     for d in year_docs:
#         d.metadata = {"Year": year}
#     doc_set[year] = year_docs
#     all_docs.extend(year_docs)

Settings.chunk_size = 512
index_set = {}

# for year in years:
#     storage_context = StorageContext.from_defaults()
#     cur_index = VectorStoreIndex.from_documents(
#         doc_set[year],
#         storage_context=storage_context,
#     )
#     index_set[year] = cur_index
#     storage_context.persist(persist_dir=f"./storage/{year}")

for year in years:
    if path.exists(f"./storage/{year}"):
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{year}"
        )
        cur_index = load_index_from_storage(
            storage_context
        )
        index_set[year] = cur_index
    else:
        print("Not exist")

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index{year}",
            description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
        )
    )
    for year in years
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-3.5-turbo"),
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
    )
)

tools = individual_query_engine_tools + [query_engine_tool]
agent = OpenAIAgent.from_tools(tools, verbose=True)

response = agent.chat("hi, I'm Kyle")
print(str(response))