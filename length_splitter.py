from langchain_text_splitters import CharacterTextSplitter


text="""Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
    git clone https://github.com/your-username/student-tracker.git"""

splitter=CharacterTextSplitter(
   chunk_size=500,        
    chunk_overlap=50,
    separator="\n"
)

result=splitter.split_text(text=text)
print(result)