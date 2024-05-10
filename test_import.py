#%%
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from qdrant_client import QdrantClient

import json
from IPython.display import display, Markdown
import json
#with open("../secrets.json", "r") as f:
#    secrets = json.load(f)




API_KEYS = secrets["keys"]


#%%
# initialization
# Get global Settings
# embedding model

model_name = "voyage-large-2"
voyage_api_key = API_KEYS["voyage-ai"]
embed_model = VoyageEmbedding(
    model_name=model_name, voyage_api_key=voyage_api_key
)
Settings.embed_model = embed_model

# llm and tokenizer
tokenizer = Anthropic().tokenizer
model_name = secrets["config"]["anthropic"]["model-sonnet"]
anthropic_api_key = API_KEYS["anthropic"]
llm = Anthropic(model=model_name,
                api_key=anthropic_api_key)
Settings.tokenizer = tokenizer
Settings.llm = llm

# chunk size
Settings.chunk_size = 1024

client = QdrantClient(
    host=secrets["config"]["qdrant-cloud"]["host"],
    port=secrets["config"]["qdrant-cloud"]["port"],
    api_key=API_KEYS["qdrant-cloud"],
)

#%%
# Set up vector DB
#with open("./collections.json","r") as f:
#    data_loader_config = json.load(f)



data_loader_config=[
    {
        "collection_name": "ASCO", 
        "description": "Treatment guidelines from American Society of Clinical Oncology."
    }, 
    {
        "collection_name": "CTR", 
        "description": "Relevant Clinical Trial information for Breast Cancer."
    }, 
    {
        "collection_name": "DRUGBANK", 
        "description": "Detailed information on Drugs for Breast Cancer"}, 
    {
        "collection_name": "ESMO", 
        "description": "Treatment guidelines from European Society of Medical Oncology."}, 
    {
        "collection_name": "JNCCN", 
        "description": "Journals from National Comprehensive Cancer Network. Includes guidelines as well. "}, 
    {
        "collection_name": "NCCN", 
        "description": "Breast cancer treatment guilines by National Comprehensive Cancer Network"}, 
    {
        "collection_name": "PUBMED", 
        "description": "Publicly accessible journals on breast cancer drugs and their targets from PubMed"}, 
    {
        "collection_name": "WIKI", 
        "description": "General information on drugs."
    }
]

query_engine_tools = []
for cfg in data_loader_config:
    collection_name = cfg['collection_name']
    description = cfg["description"]
    vector_store = QdrantVectorStore(collection_name, client=client, enable_hybrid=True, batch_size=20)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=10, 
                                         sparse_top_k=15,
                                         response_mode="tree_summarize",                                       
                                    )
    query_engine_tools.append(
            QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=collection_name,
                description=description,
            )
        )
    )

#%%
agent = ReActAgent.from_tools(
    query_engine_tools, llm=llm, verbose=True, max_iterations=20
)
response = agent.chat(
    '''
        Given the following case on breast cancer, 
        think step by step with the sources and tool names 
        the best course of action for the given patient. 
        Provide a detailed and structured output:

        Brief History: The patient is a known case of metastatic carcinoma breast, diagnosed in 2021. 
        The patient had history of lump in left breast. The patient underwent PET CT as well as USG of 
        bilateral breast followed by biopsy, suggestive of IDC Grade 2 with ER PR Positive and Her2/Neu
        Negative with Ki 67 8 to 10%. PET CT was suggestive of 2.7 x 1.8 left breast mass
        with left axillary nodes. Mammogram lymph nodes positive with bilateral lung and mediastinal 
        hilar lymph nodes positive, metastasis. The patient was started with treatment RIBOCICLIB, and ZOLADEX, TAB.
        LETROZOLE + INJ. ZOBONE. The patient completed 6 months of treatment followed by the patient was on
        follow up. The patient underwent re-evaluation with PET CT on 07.04.2022 which was suggestive of
        progression in the bone lesion as well as interval development of mild left pleural effusion, metabolically active
        pleural thickening. Hence, the patient further planned for systemic chemotherapy with INJ. NAB
        PACLITAXEL weekly + INJ. CARBOPLATIN Q 3 weekly x 12 weeks followed by reassessment with PET
        CT. Now, the patient is admitted for 1St week of systemic chemotherapy with INJ. NAB PACLITAXEL with
        no specific complaints. BRCA is Negative. Appetite is normal. Bowel and bladder habits are regular.
    '''
)

display(Markdown(str(response)))

#%%
tasks = agent.list_tasks()
task_state = tasks[-1]

# get completed steps
completed_steps = agent.get_completed_steps(task_state.task.task_id)
for idx in range(len(completed_steps)):
    print(f"Step {idx}")
    print(f"Response: {completed_steps[idx].output.response}")
    print(f"Sources: {completed_steps[idx].output.sources}")

                # %%
