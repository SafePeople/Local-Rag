# For SQLite Database
import sqlite3

# Create or connect to the database
conn = sqlite3.connect('llm_data.db')
cursor = conn.cursor()

# Create a table to store the documents and their embeddings
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        file_name TEXT,
        content TEXT)
''')

# Create a table to store user queries and the LLM responses
cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY,
        user_input TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')

conn.commit()
conn.close()

# Function to add documents to the database
def add_document(file_name: str, content: list[str]):
    conn = sqlite3.connect('llm_data.db')
    cursor = conn.cursor()
    content = '\n'.join(content)
    cursor.execute('''
        INSERT INTO documents (file_name, content) VALUES (?, ?)
    ''', (file_name, content))
    conn.commit()
    conn.close()

# Function to add user queries and LLM responses
def add_query(user_input, response):
    conn = sqlite3.connect('llm_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO queries (user_input, response) VALUES (?, ?)
    ''', (user_input, response))
    conn.commit()
    conn.close()

# Function to fetch documents from the database
def fetch_documents() -> list[str]:
    conn = sqlite3.connect('llm_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT file_name, content FROM documents
    ''')
    documents = cursor.fetchall()
    conn.close()
    return documents

def fetch_queries():
    conn = sqlite3.connect('llm_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_input, response, timestamp FROM queries ORDER BY timestamp DESC
    ''')
    queries = cursor.fetchall()
    conn.close()
    return queries

def clear_table(table_name):
    conn = sqlite3.connect('llm_data.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        DELETE FROM {table_name}
    ''')

    # Create a table to store the documents and their embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            content TEXT)
    ''')
    # Create a table to store user queries and the LLM responses
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            user_input TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()