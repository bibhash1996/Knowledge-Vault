import sys
from contextlib import nullcontext
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import List
from urllib.parse import urlsplit
from urllib.request import urlopen
import json

from sqlalchemy.orm import Session

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from app.db.vector_store import get_vector_store
from app.db.session import SessionLocal
from app.repositories.original_content import OriginalContentRepository

import dotenv
dotenv.load_dotenv() 


def partition_pdf_file(file_path: str) -> list:
    """
    Partitions a PDF file into structured elements.

    Args:
        file_path (str): A local path or HTTP(S) URL to the PDF file.

    Returns:
        list: A list of partitioned elements from the PDF file.
    """
    is_url = urlsplit(file_path).scheme in {"http", "https"}
    temporary_directory = TemporaryDirectory() if is_url else nullcontext()
    with temporary_directory as temp_dir:
        if is_url:
            local_file = Path(temp_dir) / "document.pdf"
            with urlopen(file_path, timeout=30) as response, local_file.open("wb") as output:
                copyfileobj(response, output)
            file_path = str(local_file)

        partitioned_elements = partition_pdf(
            filename=file_path,
            strategy="auto",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    
    print(f"✅ Extracted {len(partitioned_elements)} elements")
    return partitioned_elements

def create_chunks_by_title(elements:List[Element]):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(
    chunks,
    file_url: str,
    session_id: int | None = None,
    knowledge_id: int | None = None,
    db: Session | None = None,
) -> List[Document]:
    """Process all chunks with AI Summaries and tag each chunk to a session."""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents: List[Document] = []
    total_chunks = len(chunks)
    content_repository = OriginalContentRepository()
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']

        content_references = []
        if content_data['tables'] or content_data['images']:
            if db is None or session_id is None or knowledge_id is None:
                raise ValueError(
                    "Persisting tables or images requires a database session, "
                    "session_id, and knowledge_id"
                )

            entries = [
                ("table", value) for value in content_data["tables"]
            ] + [
                ("image", value) for value in content_data["images"]
            ]
            stored_content = content_repository.create_many(
                db,
                session_id=int(session_id),
                knowledge_id=int(knowledge_id),
                entries=entries,
            )
            content_references = [
                {"entity_type": row.entity_type, "id": row.id}
                for row in stored_content
            ]
        
        metadata = {
            "original_content": json.dumps({
                "raw_text": content_data['text'],
                "content_references": content_references,
            }),
            "file_url": file_url,
        }
        if session_id is not None:
            metadata["session_id"] = int(session_id)

        doc = Document(
            page_content=enhanced_content,
            metadata=metadata,
        )
        
        langchain_documents.append(doc)
    
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


def save_chunks_to_vector_store(chunks: List[Document], session_id: int) -> None:
    """Persist session-scoped document chunks into the shared vector store."""
    if not chunks:
        print("ℹ️ No chunks to save to the vector store")
        return

    try:
        vector_store = get_vector_store()
    except Exception as exc:  # pragma: no cover - depends on external Pinecone configuration
        print(f"⚠️ Skipping vector-store save for session_id={session_id}: {exc}")
        return

    tagged_documents = []
    for doc in chunks:
        metadata = dict(doc.metadata or {})
        metadata["session_id"] = int(session_id)
        tagged_documents.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    try:
        vector_store.add_documents(tagged_documents)
    except Exception as exc:  # pragma: no cover - depends on external Pinecone configuration
        print(f"⚠️ Failed to save chunks to vector store for session_id={session_id}: {exc}")
        return

    print(f"✅ Saved {len(tagged_documents)} chunks to vector store for session_id={session_id}")


def run_complete_ingestion_pipeline(
    pdf_path: str,
    session_id: int,
    knowledge_id: int | None = None,
):
    """Run the complete RAG ingestion pipeline and store session-scoped chunks."""
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Partition
    elements = partition_pdf_file(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    
    # Step 3: AI Summarisation and persistence of large content blocks.
    db = SessionLocal()
    try:
        summarised_chunks = summarise_chunks(
            chunks,
            pdf_path,
            session_id=session_id,
            knowledge_id=knowledge_id,
            db=db,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    save_chunks_to_vector_store(summarised_chunks, session_id=session_id)
    # print(summarised_chunks[0].metadata)
    print(f"✅ Completed RAG Ingestion Pipeline with {len(summarised_chunks)} summarised chunks")

if __name__ == "__main__":
    # Example usage
    # print("Partitioning PDF file...")
    # file_path = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    file_path = "https://www.sldttc.org/allpdf/21583473018.pdf"
    # partitioned_elements = partition_pdf_file(file_path)
    chunked_elements = run_complete_ingestion_pipeline(file_path, session_id=1,knowledge_id=2)
