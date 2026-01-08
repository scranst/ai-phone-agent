"""
Knowledge Base System

Stores and retrieves knowledge for AI-powered phone calls.
Supports:
- Products and pricing
- Services and procedures
- Policies and FAQs
- Custom documents

Knowledge is organized into "bases" - collections of related documents
that can be searched and used to provide context during calls.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Knowledge base storage directory
KB_DIR = Path(__file__).parent / "knowledge_bases"


def init_kb_storage():
    """Initialize knowledge base storage directory"""
    KB_DIR.mkdir(exist_ok=True)


def list_knowledge_bases() -> List[Dict[str, Any]]:
    """
    List all knowledge bases.

    Returns:
        List of knowledge base metadata dicts
    """
    init_kb_storage()

    bases = []
    for kb_dir in KB_DIR.iterdir():
        if kb_dir.is_dir():
            meta_file = kb_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        meta['id'] = kb_dir.name

                        # Count documents
                        docs_dir = kb_dir / "documents"
                        if docs_dir.exists():
                            meta['document_count'] = len(list(docs_dir.glob("*.json")))
                        else:
                            meta['document_count'] = 0

                        bases.append(meta)
                except Exception as e:
                    logger.error(f"Error loading KB metadata: {e}")

    return sorted(bases, key=lambda x: x.get('name', ''))


def create_knowledge_base(
    name: str,
    description: str = "",
    category: str = "general",
    objective_keywords: List[str] = None
) -> str:
    """
    Create a new knowledge base.

    Args:
        name: Display name for the knowledge base
        description: Description of what this KB contains
        category: Category (products, services, policies, faq, general)
        objective_keywords: Keywords that match this KB to call objectives

    Returns:
        The knowledge base ID
    """
    init_kb_storage()

    # Generate ID from name
    kb_id = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    if not kb_id:
        kb_id = "kb_" + datetime.now().strftime("%Y%m%d%H%M%S")

    kb_path = KB_DIR / kb_id

    if kb_path.exists():
        raise ValueError(f"Knowledge base '{kb_id}' already exists")

    kb_path.mkdir()
    (kb_path / "documents").mkdir()

    # Save metadata
    metadata = {
        "name": name,
        "description": description,
        "category": category,
        "objective_keywords": objective_keywords or [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    with open(kb_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created knowledge base: {name} ({kb_id})")
    return kb_id


def get_knowledge_base(kb_id: str) -> Optional[Dict[str, Any]]:
    """Get knowledge base metadata by ID"""
    kb_path = KB_DIR / kb_id
    meta_file = kb_path / "metadata.json"

    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = json.load(f)
        meta['id'] = kb_id
        return meta


def update_knowledge_base(kb_id: str, updates: Dict[str, Any]) -> bool:
    """Update knowledge base metadata"""
    kb_path = KB_DIR / kb_id
    meta_file = kb_path / "metadata.json"

    if not meta_file.exists():
        return False

    with open(meta_file) as f:
        meta = json.load(f)

    # Update allowed fields
    for field in ['name', 'description', 'category', 'objective_keywords']:
        if field in updates:
            meta[field] = updates[field]

    meta['updated_at'] = datetime.now().isoformat()

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return True


def delete_knowledge_base(kb_id: str) -> bool:
    """Delete a knowledge base and all its documents"""
    import shutil

    kb_path = KB_DIR / kb_id

    if not kb_path.exists():
        return False

    shutil.rmtree(kb_path)
    logger.info(f"Deleted knowledge base: {kb_id}")
    return True


def add_document(
    kb_id: str,
    title: str,
    content: str,
    doc_type: str = "text",
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Add a document to a knowledge base.

    Args:
        kb_id: Knowledge base ID
        title: Document title
        content: Document content (plain text)
        doc_type: Document type (text, faq, product, procedure)
        tags: Optional list of tags for filtering
        metadata: Optional additional metadata

    Returns:
        Document ID
    """
    kb_path = KB_DIR / kb_id
    docs_dir = kb_path / "documents"

    if not docs_dir.exists():
        raise ValueError(f"Knowledge base '{kb_id}' not found")

    # Generate document ID
    doc_id = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
    if not doc_id:
        doc_id = "doc_" + datetime.now().strftime("%Y%m%d%H%M%S")

    # Ensure unique ID
    base_id = doc_id
    counter = 1
    while (docs_dir / f"{doc_id}.json").exists():
        doc_id = f"{base_id}_{counter}"
        counter += 1

    document = {
        "id": doc_id,
        "title": title,
        "content": content,
        "type": doc_type,
        "tags": tags or [],
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    with open(docs_dir / f"{doc_id}.json", "w") as f:
        json.dump(document, f, indent=2)

    # Update KB modified time
    update_knowledge_base(kb_id, {})

    logger.info(f"Added document '{title}' to KB '{kb_id}'")
    return doc_id


def get_document(kb_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Get a document by ID"""
    doc_file = KB_DIR / kb_id / "documents" / f"{doc_id}.json"

    if not doc_file.exists():
        return None

    with open(doc_file) as f:
        return json.load(f)


def update_document(kb_id: str, doc_id: str, updates: Dict[str, Any]) -> bool:
    """Update a document"""
    doc_file = KB_DIR / kb_id / "documents" / f"{doc_id}.json"

    if not doc_file.exists():
        return False

    with open(doc_file) as f:
        doc = json.load(f)

    for field in ['title', 'content', 'type', 'tags', 'metadata']:
        if field in updates:
            doc[field] = updates[field]

    doc['updated_at'] = datetime.now().isoformat()

    with open(doc_file, "w") as f:
        json.dump(doc, f, indent=2)

    return True


def delete_document(kb_id: str, doc_id: str) -> bool:
    """Delete a document"""
    doc_file = KB_DIR / kb_id / "documents" / f"{doc_id}.json"

    if not doc_file.exists():
        return False

    doc_file.unlink()
    logger.info(f"Deleted document '{doc_id}' from KB '{kb_id}'")
    return True


def list_documents(
    kb_id: str,
    doc_type: str = None,
    tags: List[str] = None
) -> List[Dict[str, Any]]:
    """
    List documents in a knowledge base.

    Args:
        kb_id: Knowledge base ID
        doc_type: Optional filter by document type
        tags: Optional filter by tags (any match)

    Returns:
        List of document metadata (without full content)
    """
    docs_dir = KB_DIR / kb_id / "documents"

    if not docs_dir.exists():
        return []

    documents = []
    for doc_file in docs_dir.glob("*.json"):
        try:
            with open(doc_file) as f:
                doc = json.load(f)

            # Apply filters
            if doc_type and doc.get('type') != doc_type:
                continue
            if tags:
                doc_tags = set(doc.get('tags', []))
                if not doc_tags.intersection(tags):
                    continue

            # Return summary (without full content)
            documents.append({
                "id": doc.get('id'),
                "title": doc.get('title'),
                "type": doc.get('type'),
                "tags": doc.get('tags', []),
                "content_preview": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                "created_at": doc.get('created_at'),
                "updated_at": doc.get('updated_at')
            })
        except Exception as e:
            logger.error(f"Error loading document: {e}")

    return sorted(documents, key=lambda x: x.get('title', ''))


def search_documents(
    query: str,
    kb_ids: List[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for relevant documents across knowledge bases.

    Uses simple keyword matching. For production, consider
    using embeddings and vector search.

    Args:
        query: Search query
        kb_ids: Optional list of KB IDs to search (None = all)
        limit: Maximum results to return

    Returns:
        List of matching documents with relevance scores
    """
    if kb_ids is None:
        kb_ids = [kb['id'] for kb in list_knowledge_bases()]

    query_words = set(query.lower().split())
    results = []

    for kb_id in kb_ids:
        docs_dir = KB_DIR / kb_id / "documents"
        if not docs_dir.exists():
            continue

        for doc_file in docs_dir.glob("*.json"):
            try:
                with open(doc_file) as f:
                    doc = json.load(f)

                # Calculate simple relevance score
                content = (doc.get('title', '') + ' ' + doc.get('content', '')).lower()
                content_words = set(content.split())

                # Count matching words
                matches = len(query_words.intersection(content_words))

                # Boost exact phrase matches
                if query.lower() in content:
                    matches += 5

                if matches > 0:
                    results.append({
                        "kb_id": kb_id,
                        "doc_id": doc.get('id'),
                        "title": doc.get('title'),
                        "content": doc.get('content'),
                        "type": doc.get('type'),
                        "score": matches
                    })
            except Exception as e:
                logger.debug(f"Error searching document: {e}")

    # Sort by score and limit
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]


def get_knowledge_for_prompt(
    objective: str = None,
    context: Dict[str, Any] = None,
    kb_ids: List[str] = None,
    max_tokens: int = 2000
) -> str:
    """
    Get relevant knowledge to include in LLM prompt.

    Searches knowledge bases for content relevant to the
    call objective and returns formatted text for the prompt.

    Args:
        objective: The call objective to match against
        context: Additional context (may contain KB hints)
        kb_ids: Optional specific KBs to search
        max_tokens: Approximate max tokens to return

    Returns:
        Formatted knowledge string for prompt injection
    """
    if not objective:
        return ""

    # If no specific KB IDs, find KBs that match the objective keywords
    if kb_ids is None:
        all_kbs = list_knowledge_bases()
        objective_lower = objective.lower()
        matched_kb_ids = []

        for kb in all_kbs:
            keywords = kb.get('objective_keywords', [])
            if not keywords:
                # No keywords means always include (general KB)
                matched_kb_ids.append(kb['id'])
            else:
                # Check if any keyword matches the objective
                for keyword in keywords:
                    if keyword.lower() in objective_lower:
                        matched_kb_ids.append(kb['id'])
                        break

        kb_ids = matched_kb_ids if matched_kb_ids else [kb['id'] for kb in all_kbs]

    # Search for relevant documents
    results = search_documents(objective, kb_ids, limit=10)

    if not results:
        return ""

    # Build formatted knowledge section
    lines = ["RELEVANT KNOWLEDGE:"]
    char_count = 0
    max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars

    for doc in results:
        doc_text = f"\n### {doc['title']}\n{doc['content']}"

        if char_count + len(doc_text) > max_chars:
            # Truncate if needed
            remaining = max_chars - char_count
            if remaining > 100:
                lines.append(doc_text[:remaining] + "...")
            break
        else:
            lines.append(doc_text)
            char_count += len(doc_text)

    return "\n".join(lines)


def import_text_file(kb_id: str, file_path: str, title: str = None) -> str:
    """
    Import a text file as a document.

    Args:
        kb_id: Knowledge base ID
        file_path: Path to text file
        title: Optional title (defaults to filename)

    Returns:
        Document ID
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path) as f:
        content = f.read()

    if not title:
        title = path.stem.replace('_', ' ').replace('-', ' ').title()

    return add_document(kb_id, title, content, doc_type="text")


def import_faq(kb_id: str, faq_data: List[Dict[str, str]]) -> List[str]:
    """
    Import FAQ data as documents.

    Args:
        kb_id: Knowledge base ID
        faq_data: List of {"question": "...", "answer": "..."}

    Returns:
        List of document IDs
    """
    doc_ids = []

    for item in faq_data:
        question = item.get('question', '')
        answer = item.get('answer', '')

        if question and answer:
            content = f"Q: {question}\n\nA: {answer}"
            doc_id = add_document(
                kb_id,
                question[:100],  # Title is question (truncated)
                content,
                doc_type="faq"
            )
            doc_ids.append(doc_id)

    return doc_ids


def import_products(
    kb_id: str,
    products: List[Dict[str, Any]]
) -> List[str]:
    """
    Import product catalog.

    Args:
        kb_id: Knowledge base ID
        products: List of product dicts with name, description, price, etc.

    Returns:
        List of document IDs
    """
    doc_ids = []

    for product in products:
        name = product.get('name', 'Unknown Product')

        # Build product content
        lines = [f"Product: {name}"]

        if product.get('description'):
            lines.append(f"Description: {product['description']}")
        if product.get('price'):
            lines.append(f"Price: ${product['price']}")
        if product.get('sku'):
            lines.append(f"SKU: {product['sku']}")
        if product.get('features'):
            lines.append("Features:")
            for feature in product['features']:
                lines.append(f"  - {feature}")
        if product.get('specs'):
            lines.append("Specifications:")
            for key, value in product['specs'].items():
                lines.append(f"  - {key}: {value}")

        content = "\n".join(lines)

        doc_id = add_document(
            kb_id,
            name,
            content,
            doc_type="product",
            metadata=product
        )
        doc_ids.append(doc_id)

    return doc_ids


# Convenience function for quick setup
def create_sample_knowledge_base():
    """Create a sample knowledge base with example content"""
    try:
        kb_id = create_knowledge_base(
            "Sample Business Info",
            "Sample knowledge base with business information",
            "general"
        )

        # Add some sample documents
        add_document(
            kb_id,
            "Business Hours",
            "We are open Monday through Friday, 9 AM to 5 PM. "
            "Saturday hours are 10 AM to 2 PM. We are closed on Sundays and major holidays.",
            doc_type="procedure"
        )

        add_document(
            kb_id,
            "Return Policy",
            "Items can be returned within 30 days of purchase with receipt. "
            "All returns must be in original condition. Refunds are processed "
            "within 5-7 business days. Store credit is available for items "
            "returned after 30 days.",
            doc_type="policy"
        )

        add_document(
            kb_id,
            "Contact Information",
            "Email: support@example.com\n"
            "Phone: 1-800-EXAMPLE\n"
            "Address: 123 Main Street, Suite 100, Anytown, USA 12345",
            doc_type="text"
        )

        logger.info(f"Created sample knowledge base: {kb_id}")
        return kb_id

    except ValueError:
        # Already exists
        return "sample_business_info"


if __name__ == "__main__":
    # Test the knowledge base system
    logging.basicConfig(level=logging.INFO)

    print("Testing Knowledge Base System")
    print("=" * 50)

    # Create sample KB
    kb_id = create_sample_knowledge_base()
    print(f"\nCreated/Found KB: {kb_id}")

    # List KBs
    print("\nKnowledge Bases:")
    for kb in list_knowledge_bases():
        print(f"  - {kb['name']} ({kb['id']}): {kb['document_count']} docs")

    # List documents
    print(f"\nDocuments in {kb_id}:")
    for doc in list_documents(kb_id):
        print(f"  - {doc['title']} ({doc['type']})")

    # Search test
    print("\nSearch for 'return':")
    results = search_documents("return policy refund")
    for r in results:
        print(f"  - {r['title']} (score: {r['score']})")

    # Get knowledge for prompt
    print("\nKnowledge for prompt (objective: 'ask about return policy'):")
    knowledge = get_knowledge_for_prompt("What is your return policy?")
    print(knowledge[:500] + "..." if len(knowledge) > 500 else knowledge)
