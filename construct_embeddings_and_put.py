import pyodbc
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import dotenv_values
from bs4 import BeautifulSoup


MAIN_SOLUTIONS_QUERY = """
--- Findaso Solution content reconstruction query

with pricing_option_cte as (
SELECT a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId, STRING_AGG(c.title, ', ') as pricing_options
from findasoc_MainDB.findasomaster.ReadySolutionLanguage a  
-- pricing options 
FULL JOIN findasoc_MainDB.dbo.ReadySolutionPricingOptionLinks b on a.ReadySolutionId = b.[ReadySolutionId]
FULL JOIN [findasoc_MainDB].ReadySolution.PricingOptions c on b.PricingOptionId = c.Id
GROUP BY a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId
),
targeted_customers as (
SELECT  a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId,  STRING_AGG(e.Title, ',') as customer_types 
from findasoc_MainDB.findasomaster.ReadySolutionLanguage a  
-- targeted customers
FULL JOIN findasoc_MainDB.dbo.ReadySolutionCustomerTypeLinks d on d.ReadySolutionId = a.ReadySolutionId
FULL JOIN findasoc_MainDB.ReadySolution.CustomerTypes e on e.Id = d.CustomerTypeId
GROUP BY a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId
),
solution_types as (
SELECT a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId,  STRING_AGG(g.Title, ',') as solution_types
from findasoc_MainDB.findasomaster.ReadySolutionLanguage a  
-- solution type
FULL JOIN findasoc_MainDB.dbo.ReadySolutionTypeLinks f on f.ReadySolutionId = a.ReadySolutionId
FULL JOIN findasoc_MainDB.ReadySolution.ReadySolutionTypes g on g.Id = f.ReadySolutionTypeId
GROUP BY a.Id, a.title, a.Abstract, a.Description, a.keywords, a.ReadySolutionId
),
active_sales_centers as (
SELECT ReadySolutionId, STRING_AGG(b.Country, ',') as sales_centers from findasoc_MainDB.[ReadySolution].[SalesPartnerLinks] a
join findasoc_MainDB.[Company].[Company] b on a.SalesPartnerId = b.Id
GROUP BY ReadySolutionId
),
origin_countries as (
select a.ReadySolutionId, c.Country from findasoc_MainDB.findasomaster.ReadySolutionLanguage a
JOIN findasoc_MainDB.[ReadySolution].[ReadySolutions] b on a.ReadySolutionId = b.Id
JOIN findasoc_MainDB.[Company].[Company] c on b.CompanyId = c.Id
)
select distinct a.Id, a.title, a.Abstract, a.Description, a.Keywords, a.pricing_options, b.customer_types, c.solution_types, e.Country as country , d.sales_centers, CONCAT('https://www.findaso.com/ready-solution/',a.ReadySolutionId) as link from pricing_option_cte a
FULL join targeted_customers b on a.Id = b.Id
FULL join solution_types c on a.Id = c.Id
FULL JOIN active_sales_centers d on a.ReadySolutionId = d.ReadySolutionId
FULL JOIN origin_countries e on e.ReadySolutionId = a.ReadySolutionId
where a.Id is not null
order by Id

"""

config = dotenv_values(".env")


# MS SQL Database configuration
connection_string = config["DB_URI"]

pc = Pinecone(api_key=config["PINECONE_API_KEY"])

# Pinecone index configuration
index_name = "finda"
if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(index_name, dimension=1536,
                      spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                        )
                    )
index = pc.Index(index_name)

# Function to connect to the database and retrieve blog posts
def fetch_solutions():
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute(MAIN_SOLUTIONS_QUERY)
    posts = cursor.fetchall()
    connection.close()
    return posts

# Function to generate Markdown content
def generate_markdown(title, abstract, description, keywords, pricing_options, customer_types, solution_types, country, sales_centers, link):
    abstract_md = None
    description_md = None
    
    if abstract: 
        abstract = BeautifulSoup(abstract) if abstract else None
    if description:
        description = BeautifulSoup(description) if description else None
    abstract_md = '\n'.join(i.text for i in abstract.find_all("p")) if abstract else "None"
    description_md = '\n'.join(i.text for i in description.find_all("p")) if description else "None"
    markdown_content = f"""# {title}

## Abstract
{abstract_md}

## Description 
{description_md}

## Keywords
{keywords}

## Pricing Options
{pricing_options}

## Customer Types
{customer_types}

## Solution Types
{solution_types}

## Country
{country}

## Sales Centers 
{sales_centers}

## Link
{link}

"""
    return markdown_content

# Initialize Markdown splitter with a large chunk size to keep each post as a single chunk
markdown_splitter = MarkdownTextSplitter(chunk_size=10000, chunk_overlap=0)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=config["OPENAI_KEY"])  # Ensure model dimension matches Pinecone index

# Fetch posts from the database
solutions = fetch_solutions()

# Process each post
for idx, (id, title, abstract, description, keywords, pricing_options, customer_types, solution_types, country, sales_centers, link) in enumerate(solutions, start=1):
    markdown_content = generate_markdown(title, abstract, description, keywords, pricing_options, customer_types, solution_types, country, sales_centers, link)
    split_content = markdown_splitter.split_text(markdown_content)
  
    # Only one chunk should exist if the chunk size is large enough
    for part_idx, part in enumerate(split_content, start=1):
        # Create unique ID for Pinecone entry
        vector_id = f"solution_{id}"
        
        # Generate embedding for the single chunk
        embedding = embedder.embed_query(part)
        
        # Upload to Pinecone
        index.upsert([(vector_id, embedding, {"title": title, "id": id, "text": part})])

        print(f"Uploaded embedding for '{title}' - part {part_idx}")

print("All embeddings generated and uploaded to Pinecone!")
