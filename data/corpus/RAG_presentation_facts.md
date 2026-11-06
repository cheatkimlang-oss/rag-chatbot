# RAG presentation: facts

Restated from "RAG presentation.pdf" (Project Practicum I, Group 1) as one fact per sentence.

## About the RAG project and team

The RAG project is a Project Practicum I presentation by Group 1 at the Royal University of Phnom Penh.
The RAG project is supervised by lecturer Chap Chanpiseth.
The researchers of Group 1 are Ngoun Nuongswayumphou and Cheat Kimlang.

## What RAG is

Retrieval-Augmented Generation (RAG) is an approach in which a language model consults external verified documents before generating output, instead of answering from memory.
The four stages of RAG are Indexing, Retrieval, Augmentation, and Generation.
In the Indexing stage, raw documents are parsed and stored as vector embeddings.
In the Retrieval stage, a semantic similarity search fetches the top-k matching chunks.
In the Augmentation stage, the user prompt is merged with the retrieved excerpts into an enriched prompt.
In the Generation stage, the LLM generates a grounded answer backed by the provided data.
RAG reduces hallucination by forcing responses to cite retrieved passages, which removes fabricated knowledge and outdated facts.
A RAG knowledge base is updated by updating its index files, for example by uploading new files, without costly model retraining.
RAG does not need model retraining or fine-tuning to add new data such as next semester's timetables and courses.

## The original RAG paper

The original RAG paper was written by Patrick Lewis et al. at Facebook AI Research (Meta AI), with University College London and New York University.
The RAG paper was published at NeurIPS 2020, the 34th Conference on Neural Information Processing Systems.
The title of the RAG paper is "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks".
The institutions that created RAG are Meta AI (Facebook AI Research), University College London (UCL), and New York University (NYU).

## Generative AI versus RAG

The difference between generative AI and RAG: standalone generative AI answers from its internal training weights only, while RAG answers from curated external documents.
Standalone generative AI has no access to private university data, while RAG integrates private handbooks and records.
Standalone generative AI knowledge is fixed at its training cutoff, while RAG updates instantly when new files are added.
Standalone generative AI is prone to convincing hallucinations, while RAG answers strictly from retrieved context.

## What RAG is used for

RAG is mainly used for three things: private data question answering, helpdesks and support, and academic assistants.
For private data QA, RAG queries confidential knowledge bases, PDF libraries, legal archives, and internal wikis.
For helpdesks, RAG powers customer service agents that answer policy and administrative questions with citations.
As an academic assistant, RAG indexes slide decks, textbooks, and syllabus rules into 24/7 AI tutors for students.

## Strengths of RAG

The three strengths of RAG are multi-modal data intake, configurable retrieval, and grounded, verifiable answers.
RAG can ingest PDF, HTML, Markdown, and DOCX files, as well as university websites, lecture notes, slides, and video transcripts.
RAG offers configurable retrieval: control over chunk sizes, overlap, top-k ranking, and similarity algorithms such as cosine and HNSW.
When the answer is missing from its documents, the system says I do not know instead of making something up.

## The RUPP student chatbot proposal

The problem the RUPP student chatbot solves is that students search disconnected PDFs, Facebook notice boards, and separate faculty sites to find major lists and schedules.
The proposed solution is a centralized RAG agent, updated each semester by uploading new handbook PDFs, that answers with verified campus data.
The planned tech stack for the RUPP chatbot is LangChain as the orchestrator, Qdrant as the vector store, and Google Gemini as the LLM.
The LLM planned for the RUPP chatbot is Google Gemini.
The vector store planned for the RUPP chatbot is Qdrant.
The RUPP chatbot aims to give more than 10,000 RUPP students instant, validated answers from official university policies.
The four steps of the engineering pipeline are data intake, preparation, retrieval, and LLM synthesis.
In data intake, the team collects official RUPP PDF catalogs, timetable sheets, and degree requirements.
In preparation, documents are chunked, embedded as dense vectors, and stored in a vector database.
In retrieval, the query is encoded into a vector and a cosine similarity search returns the top-k fragments.
In LLM synthesis, Gemini is prompted with the retrieved chunks to produce grounded answers.
