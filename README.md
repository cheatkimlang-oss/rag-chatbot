# RUPP New Student Assistant

A student-built Telegram and browser assistant for RUPP curriculum and campus questions. It uses saved curriculum records for exact lookups and local search and Gemma 4 for supported general questions.

[Workflow diagram](docs/RUPP_Student_Assistant_Diagram.png) | [Step-by-step handover guide (PDF)](docs/RUPP_Student_Assistant_From_Zero_To_Working_Bot.pdf)

## Project status

This is a working local prototype for new students in ITE, DSE, and AIE. It covers program and course questions, selected campus information, and source links. The browser and Telegram bot share the same answer service.

Before calling the enrollment assistant ready for student use:

- Verify fees, deadlines, required documents, and application steps for each supported program against the current intake announcement. Until then, the bot gives students the department contact and its source.
- Test complete answers and citations on a larger set of real student questions, including short English, Khmer, and questions the bot should decline. The current retrieval benchmark checks source ranking and whether the search returns no match.
- Have another person run the setup from a clean checkout and test the bot in Telegram. The current bot runs while the host computer stays awake and online. Hosting is a separate milestone if students need access while that computer is off.

## Quick start after cloning

Open PowerShell in the cloned folder. The official curriculum snapshot and curated source notes are included. Build the local indexes before starting the app.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -X utf8 .\src\ingest.py --collection official
    .\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v

With Ollama installed and running, enable hybrid search and generated answers:

    ollama pull gemma4:e2b
    ollama pull bge-m3:567m
    .\.venv\Scripts\python.exe -X utf8 .\src\hybrid.py

Run the local readiness check, then open a terminal chat or double-click **Start Telegram Bot.cmd** for Telegram:

    .\.venv\Scripts\python.exe -X utf8 .\src\telegram_bot.py --check
    .\.venv\Scripts\python.exe -X utf8 .\src\answer.py --collection official

Use your own BotFather token at the hidden prompt. The virtual environment, generated indexes, and supplied lecture PDF are kept locally. The offline suite skips the optional lecture check when its index is absent. To run that check and the PDF exercises, put an authorized copy of the simulation lecture in `data/practice/` and build the practice index as described below. The absolute Windows paths later in this guide refer to the original project folder.


## Folder guide

- src/ holds the Python code. Start with student_assistant.py.
- data/practice/ is for the simulation lecture PDF.
- data/official/ is for RUPP enrollment and orientation documents you are allowed to use.
- evaluation/ holds questions, expected answers, and source pages.

## Run the PDF reader

Run these commands in a Command Prompt terminal in VS Code. The Run Code button does not pass the PDF path.

    cd /d "D:\Practicum Project\rupp-student-assistant"
    python -m venv .venv

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -X utf8 src/student_assistant.py "data/practice/01_Intro to Simulation and Decision Analytics.pdf" --page 1

Look for the cafeteria example on page 1.

## Index a folder

Ingestion reads PDFs and curated .web.json notes directly inside the selected folder and splits pages into overlapping passages. Build the practice index with:
    .\.venv\Scripts\python.exe src/ingest.py --collection practice

The output is index/passages.jsonl. Each record has a filename, page number, and passage text.

## Search the practice passages

    .\.venv\Scripts\python.exe -X utf8 src/retrieve.py "Why does the cafeteria use simulation to decide how many meals to prepare?"

The script ranks passages by keyword overlap and prints the highest-ranked passage with its filename and page. Try the enrollment question in evaluation/questions.md to see the no-match response. Keyword overlap is a baseline and does not verify whether a passage answers the question.
## Generate a short answer with Gemma 4

Install and start [Ollama for Windows](https://ollama.com/download/windows), then run these commands from the project folder in Command Prompt:

```cmd
ollama pull gemma4:e2b
.\.venv\Scripts\python.exe -X utf8 src\answer.py "Why does the cafeteria use simulation to decide how many meals to prepare?"
.\.venv\Scripts\python.exe -X utf8 src\answer.py "What documents do I need to bring for enrollment?"
```

The first command downloads the model once. For a matching question, the script sends the passages from the best matching PDF page to local Ollama and adds the page citation. If Ollama is unavailable, it shows the retrieved passage. With no match, the script returns immediately without calling Ollama.

## Start an interactive practice chat

From the project folder in Command Prompt, run:

    .\.venv\Scripts\python.exe -X utf8 src\answer.py

Ask several questions in one session. Use complete questions since each one is searched independently. Type /exit to close the chat. The current index contains the simulation lecture, so use it to test the workflow while waiting for official RUPP sources.

## Use official RUPP documents

The official collection already includes RUPP web notes. You can also put official PDFs directly inside data/official. To rebuild the passage index and open chat, run:

    .\.venv\Scripts\python.exe -X utf8 src\ingest.py --collection official
    .\.venv\Scripts\python.exe -X utf8 src\answer.py --collection official

Official chat searches index/official.jsonl. Practice chat searches index/passages.jsonl. Rebuild the appropriate index whenever its PDFs change, then restart the chat to load the updated index. An empty folder or an unreadable PDF stops the rebuild and preserves the previous index.

### RUPP undergraduate programs and curriculum

Official chat loads a local curriculum snapshot for Information Technology Engineering (ITE), Data Science and Engineering (DSE), and Artificial Intelligence Engineering (AIE). Exact lookups display the published Credit and Hours values without asking Gemma to rewrite them. Each curriculum answer cites the corresponding program page.

- [ITE department and contacts](https://www.rupp.edu.kh/academics/faculties_and_institutes/faculty_of_engineering/department_of_information_technology_engineering)
- [ITE curriculum](https://www.rupp.edu.kh/academics/faculties_and_institutes/faculty_of_engineering/department_of_information_technology_engineering/information_technology_engineering_bachelor_degree)
- [DSE curriculum](https://www.rupp.edu.kh/academics/faculties_and_institutes/faculty_of_engineering/department_of_information_technology_engineering/data_science_and_engineering_bachelor_degree)
- [AIE curriculum (Prospect)](https://www.rupp.edu.kh/academics/faculties_and_institutes/faculty_of_engineering/department_of_information_technology_engineering/aie)

Run from the project folder in PowerShell:

```powershell
Set-Location 'D:/Practicum Project/rupp-student-assistant'
./.venv/Scripts/python.exe -X utf8 src/answer.py --collection official
```

Try asking:

- What undergraduate majors does the ITE Department offer?
- What subjects do AIE students study in Year 3, Semester 2?
- How many credits is Deep Learning in AIE?
- Does AIE include an internship?
- Compare ITE, DSE, and AIE.
- What is the tuition fee for AIE?

#### Refresh the sources

```powershell
./.venv/Scripts/python.exe -X utf8 src/refresh_curriculum.py
./.venv/Scripts/python.exe -X utf8 src/ingest.py --collection official
./.venv/Scripts/python.exe -B -X utf8 -m unittest discover -s tests -v
```

The refresh reads the four official HTML pages, validates four years and both semesters per degree, and atomically replaces data/official/curricula.json after validation. It prints course and flagged-row counts. If the public pages are inaccessible, save the official HTML as department.html, ite.html, dse.html, and aie.html in a folder and pass that folder with --html-dir. Restart chat after a refresh. Update any separately curated .web.json notes by checking their cited pages, then rebuild the passage index. The practice index remains independent.

#### Known limits

- Fees, deadlines, scholarship eligibility, and admission requirements are unavailable in the verified source set. The assistant refers those questions to the department and cites its contact page.
- AIE's Prospect introduction names DSE and its Introduction tab has placeholder text. The DSE page contains unrelated Environmental Engineering text. Program summaries use the relevant course tables and verified ITE program introduction.
- Some AIE Hours cells have malformed notation or a prefix different from the Credit column. Original text is preserved and affected rows are flagged. Blank and partial hour breakdowns remain incomplete. ITE's fourth-year curriculum includes distinct options.
- Exact questions are answered from local records. General official questions use hybrid search when available and fall back to keyword search, which can miss paraphrases. Gemma 4 can write answers from retrieved passages. Khmer support covers selected degree terms, numerals, year and semester patterns. Answers and course titles are in English.

The official chat shows short example questions at startup. Enter a number to ask one, type your own question, or use /examples to show the list again.

## Open the local web chat

From PowerShell:

    Set-Location 'D:\Practicum Project\rupp-student-assistant'
    .\.venv\Scripts\python.exe -X utf8 .\src\web_app.py

Open http://127.0.0.1:8765 in your browser. Click a suggested question or type your own. Keep PowerShell open while using the page. Press Ctrl+C in that terminal to stop it. If the port is already in use, add --port 8766 and use the URL printed by the server. After refreshing source files, restart the server to load them.

## Hybrid search for official RUPP sources

The official pipeline starts with curated web notes and approved PDFs. Ingestion validates them and preserves URLs or PDF page numbers. LangChain splits the passages, Ollama BGE-M3 creates dense vectors, and local Qdrant combines meaning-based and word-based search with reciprocal rank fusion. Results are grouped by source. The word-based branch uses a local BM25-style scorer, not BGE-M3's learned sparse weights.

Exact curriculum, duration and contact questions still use the verified records directly. Unverified enrollment facts still get an explicit no-answer response. For general questions, the chat searches first and sends the best source to Gemma. The search can abstain before generation. Program-specific queries limit retrieval to the named degree's source.

From the project folder in PowerShell:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    ollama pull bge-m3:567m
    .\.venv\Scripts\python.exe -X utf8 .\src\ingest.py --collection official
    .\.venv\Scripts\python.exe -X utf8 .\src\hybrid.py
    .\.venv\Scripts\python.exe -X utf8 .\src\evaluate_retrieval.py
    .\.venv\Scripts\python.exe -X utf8 .\src\web_app.py

Ollama must be running for semantic search and Gemma answers. The vector snapshot is stored separately from the passage indexes. Rebuild it when official source notes or PDFs change, then restart the chat. An absent or stale vector snapshot, or an unavailable embedding service, falls back to keyword search. The practice collection stays on its original path.

The evaluator compares the top source and no-match behavior on a small set of labeled student questions. It measures retrieval rather than answer faithfulness and does not establish accuracy for every topic or language.

## Run the Telegram bot

On Windows, open Ollama and double-click Start Telegram Bot.cmd in the project folder. It uses the project's Python environment and opens an interactive token prompt.

The Telegram bot uses python-telegram-bot 22.8 and the same official sources and answer service as the browser chat. In a private chat, /start or /examples shows eight tappable questions. Students can also type their own.

1. In Telegram, open https://t.me/BotFather and use /newbot to create a bot. Keep its token private.
2. In PowerShell, run the commands below. Paste the token only into the hidden terminal prompt. You may also supply TELEGRAM_BOT_TOKEN through the environment for an unattended run.
3. Open the new bot in Telegram and send /start. Keep PowerShell open while the bot runs. Press Ctrl+C to stop it.

    Set-Location 'D:\Practicum Project\rupp-student-assistant'
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -X utf8 .\src\telegram_bot.py

To check the installed version, local sources, and search index without contacting Telegram:

    .\.venv\Scripts\python.exe -X utf8 .\src\telegram_bot.py --check

### Test in a Telegram group

Leave Telegram group privacy mode enabled and add the bot as a regular member. In the group, send `/examples@RuppPracticumAssistantBot` and tap a suggested question, or send `/ask@RuppPracticumAssistantBot How many years for DSE?`. The bot handles those commands and button taps in groups. Ordinary group messages are ignored. In private chat, students can still type questions directly.

The bot runs on this laptop using long polling. General generated answers need local Ollama running. Exact curriculum and contact answers use local records.

Run only one Telegram bot session for this project. A second local launch stops before it asks for the token. The lock clears automatically when the bot exits. If Telegram still reports a polling conflict, stop other copies using the same bot token, including on another device or host.
